"""Optuna HPO for ordinal LightGBM ranking (one query per video)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from dataclasses import asdict, replace

import optuna
import numpy as np

from ..config_loader import Config
from ..ml.model import LightGBMConfig
from ..ml.lightgbm_training import train_lightgbm_lovo
from ..dataset.manifest import load_manifest
from .lightgbm_training import (
    _lightgbm_config,
    _section,
    prepare_lightgbm_manifest_data,
)

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "objective": {"choices": ["lambdarank", "rank_xendcg"]},
    "num_leaves": {"choices": [15, 31, 63]},
    "learning_rate": {"low": 0.01, "high": 0.1, "log": True},
    "max_depth": {"choices": [3, 4, 5, 6]},
    "min_data_in_leaf": {"choices": [10, 20, 50, 100]},
    "feature_fraction": {"low": 0.6, "high": 1.0},
    "bagging_fraction": {"low": 0.6, "high": 1.0},
    "lambda_l1": {"low": 1e-4, "high": 10.0, "log": True},
    "lambda_l2": {"low": 1e-4, "high": 10.0, "log": True},
    "min_split_gain": {"low": 0.0, "high": 0.2},
}


def _validated_search_space(settings: Any) -> dict[str, dict[str, Any]]:
    section = _section(settings, "lightgbm_hpo")
    raw = section.get("search_space", {})
    if isinstance(raw, Config):
        raw = raw.to_dict()
    if not isinstance(raw, dict):
        raise ValueError("lightgbm_hpo.search_space must be a mapping.")

    # User entries override the defaults; omitted parameters remain tunable.
    space: dict[str, dict[str, Any]] = {
        name: dict(spec) for name, spec in DEFAULT_SEARCH_SPACE.items()
    }
    # Accept common native LightGBM aliases in YAML, then normalize them to
    # the canonical LightGBMConfig field names used by dataclasses.replace().
    aliases = {
        "bagging_freq": "bagging_frequency",
        "reg_alpha": "lambda_l1",
        "reg_lambda": "lambda_l2",
    }
    normalized_raw: dict[str, Any] = {}
    for supplied_name, value in raw.items():
        name = aliases.get(supplied_name, supplied_name)
        if name in normalized_raw:
            raise ValueError(
                f"Multiple HPO search-space keys map to {name!r}; "
                "use only one canonical name or alias."
            )
        normalized_raw[name] = value
    raw = normalized_raw

    allowed = set(LightGBMConfig.__dataclass_fields__) - {
        "task", "early_stopping_rounds", "ndcg_eval_at", "random_seed", "verbosity"
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown LightGBM HPO search parameter(s): {unknown}.")

    for name, value in raw.items():
        if isinstance(value, Config):
            value = value.to_dict()
        # A scalar is shorthand for a fixed parameter value, e.g.
        # bagging_freq: 1. Mapping forms define a search space explicitly.
        if not isinstance(value, dict):
            spec = {"fixed": value}
        elif "fixed" in value:
            spec = {"fixed": value["fixed"]}
        elif "choices" in value:
            choices = value["choices"]
            if not isinstance(choices, (tuple, list)) or not choices:
                raise ValueError(f"Search choices for {name!r} must be a non-empty list.")
            spec = {"choices": list(choices)}
        else:
            if "low" not in value or "high" not in value:
                raise ValueError(f"Search parameter {name!r} requires low/high or choices/fixed.")
            low, high = float(value["low"]), float(value["high"])
            if not np.isfinite([low, high]).all() or low > high:
                raise ValueError(f"Invalid low/high bounds for {name!r}.")
            spec = {"low": low, "high": high, "log": bool(value.get("log", False))}
            if "step" in value:
                spec["step"] = float(value["step"])
        space[name] = spec

    expected_objectives = ["lambdarank", "rank_xendcg"]
    objective_spec = space.get("objective")
    if objective_spec is None:
        space["objective"] = {"choices": expected_objectives}
    elif objective_spec.get("choices") != expected_objectives:
        raise ValueError(
            "lightgbm_hpo.search_space.objective must have exactly these choices, "
            "in this order: ['lambdarank', 'rank_xendcg']."
        )
    return space


def _suggest(trial: optuna.Trial, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in space.items():
        if "fixed" in spec:
            params[name] = spec["fixed"]
        elif "choices" in spec:
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif "step" in spec:
            low, high, step = spec["low"], spec["high"], spec["step"]
            if float(low).is_integer() and float(high).is_integer() and float(step).is_integer():
                params[name] = trial.suggest_int(name, int(low), int(high), step=int(step))
            else:
                params[name] = trial.suggest_float(name, low, high, step=step)
        else:
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec["log"]
            )
    return params


def run_lightgbm_hpo(manifest_path: str | Path, settings: Any) -> dict[str, Any]:
    """Tune LightGBM hyperparameters by mean per-video MTB-3 Segment Recall@5.

    Each video is a separate query for LambdaRank and for evaluation. Optuna
    maximizes the unweighted mean Segment Recall@5 across held-out LOVO groups.
    """
    section = _section(settings, "lightgbm_hpo")
    n_trials = section.get("n_trials", 30)
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError("lightgbm_hpo.n_trials must be a positive integer.")
    study_name = section.get("study_name", "lightgbm_ranking_hpo")
    if not isinstance(study_name, str) or not study_name.strip():
        raise ValueError("lightgbm_hpo.study_name must be a non-empty string.")
    # Do not mix historical NDCG trial values with the new event-recall objective.
    if not study_name.endswith("_segment_recall_at_5"):
        study_name = f"{study_name}_segment_recall_at_5"
    progress = section.get("progress", True)
    if not isinstance(progress, bool):
        raise ValueError("lightgbm_hpo.progress must be true or false.")

    def report(message: str) -> None:
        # HPO progress is deliberately independent of logging.level, so WARNING
        # can continue to suppress noisy cache diagnostics.
        if progress:
            print(f"[LightGBM HPO] {message}", flush=True)

    report(f"Preparing manifest data from {manifest_path}.")

    base_config = _lightgbm_config(settings)
    search_space = _validated_search_space(settings)

    # Proxy groups are an explicit HPO-only subset: filter before feature
    # preparation so non-proxy videos are not decoded or extracted.
    proxy_value = section.get("proxy_test_groups")
    selected_records = None
    if proxy_value is not None:
        if isinstance(proxy_value, Config):
            proxy_value = proxy_value.to_dict()
        if not isinstance(proxy_value, (tuple, list)) or not proxy_value:
            raise ValueError("lightgbm_hpo.proxy_test_groups must be a non-empty list of group IDs.")
        proxy_groups: list[str] = []
        for item in proxy_value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("lightgbm_hpo.proxy_test_groups must contain non-empty string group IDs.")
            group = item.strip()
            if group in proxy_groups:
                raise ValueError(f"lightgbm_hpo.proxy_test_groups contains duplicate group ID {group!r}.")
            proxy_groups.append(group)
        report(f"Loading manifest to select proxy groups: {', '.join(proxy_groups)}.")
        all_records = load_manifest(manifest_path, verify_files=True)
        available_groups = {record.group_id for record in all_records}
        unknown_groups = sorted(set(proxy_groups) - available_groups)
        if unknown_groups:
            raise ValueError(
                "lightgbm_hpo.proxy_test_groups not found in manifest: "
                + ", ".join(unknown_groups)
            )
        selected_records = tuple(record for record in all_records if record.group_id in proxy_groups)
        counts = {group: sum(record.group_id == group for record in selected_records) for group in proxy_groups}
        report("Proxy subset selected: " + ", ".join(f"{group}={counts[group]} videos" for group in proxy_groups) + f"; total={len(selected_records)} videos.")
        if len({record.group_id for record in selected_records}) < 2:
            raise ValueError("Proxy-group HPO requires at least two groups for LOVO evaluation.")
    else:
        report("No proxy_test_groups configured; using all manifest groups.")

    prepared, group_by_video, data_config = prepare_lightgbm_manifest_data(
        manifest_path, settings, records=selected_records, progress_callback=report
    )
    report(
        f"Prepared {len(prepared)} videos across "
        f"{len(set(group_by_video.values()))} groups; opening study database."
    )
    db_path = Path(section.get("storage_path", "data/db/lightgbm_hpo.db")).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_path.as_posix()}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=base_config.random_seed),
    )

    trial_counter = 0

    def objective(trial: optuna.Trial) -> float:
        nonlocal trial_counter
        trial_counter += 1
        params = _suggest(trial, search_space)
        report(
            f"Trial {trial_counter}/{n_trials} started "
            f"(study trial #{trial.number}; objective={params.get('objective')})."
        )
        trial_config = replace(base_config, **params)
        folds = train_lightgbm_lovo(
            prepared,
            group_by_video,
            config=trial_config,
            data_config={
                **data_config,
                "manifest": str(Path(manifest_path).expanduser().resolve()),
                "ranking_hpo_trial": trial.number,
            },
        )
        per_video_scores: list[float] = []
        per_fold: dict[str, float] = {}
        for fold in folds:
            metrics = fold.summary.get("validation_metrics", {})
            score = metrics.get("segment_recall_at_5")
            if score is not None and np.isfinite(float(score)):
                per_fold[fold.held_out_group] = float(score)
                per_video = metrics.get("per_video", {})
                per_video_scores.extend(
                    float(item["recall"])
                    for item in per_video.values()
                    if np.isfinite(float(item["recall"]))
                )
        if not per_video_scores:
            raise ValueError(
                f"Trial {trial.number} produced no validation videos with class-3 events."
            )
        mean_recall = float(np.mean(per_video_scores))
        trial.set_user_attr("metric", "segment_recall_at_5")
        trial.set_user_attr("mean_segment_recall_at_5", mean_recall)
        trial.set_user_attr("fold_segment_recall_at_5", per_fold)
        trial.set_user_attr("evaluated_video_count", len(per_video_scores))
        trial.set_user_attr("evaluated_folds", sorted(per_fold))
        logger.info(
            "LightGBM HPO trial %d: objective=%s, mean Segment Recall@5=%.5f, folds=%s",
            trial.number, params.get("objective", base_config.objective),
            mean_recall, per_fold,
        )
        report(
            f"Trial {trial_counter}/{n_trials} finished: "
            f"mean Segment Recall@5={mean_recall:.5f}."
        )
        return mean_recall

    logger.info(
        "Starting LightGBM HPO study %r: %d new trials, objectives=%s, groups=%s",
        study_name, n_trials, search_space["objective"]["choices"],
        sorted(set(group_by_video.values())),
    )
    report(
        f"Study '{study_name}': running {n_trials} new trials "
        f"({len(study.trials)} trial records already in the database); "
        "optimizing mean Segment Recall@5. "
        "A trial may take time while its LOVO folds run."
    )
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not completed:
        raise RuntimeError(f"LightGBM HPO study {study_name!r} has no completed trials.")
    best = study.best_trial

    output_dir = Path(section.get("output_dir", "output/hpo")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "study_name": study_name,
        "objective": "mean_per_video_segment_recall_at_5",
        "metric": "segment_recall_at_5",
        "direction": "maximize",
        "evaluated_groups": sorted(set(group_by_video.values())),
        "proxy_test_groups": sorted(set(group_by_video.values())) if proxy_value is not None else None,
        "completed_trials": len(completed),
        "best_trial_number": best.number,
        "best_value": best.value,
        "best_params": asdict(replace(base_config, **best.params)),
        "best_fold_segment_recall_at_5": best.user_attrs.get("fold_segment_recall_at_5", {}),
        "trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "fold_segment_recall_at_5": trial.user_attrs.get("fold_segment_recall_at_5", {}),
            }
            for trial in completed
        ],
    }
    output_path = output_dir / f"{study_name}_results.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "LightGBM HPO complete: best trial %d, objective=%s, mean Segment Recall@5=%.5f; results: %s",
        best.number, best.params.get("objective", base_config.objective),
        float(best.value), output_path,
    )
    report(
        f"Finished. Best trial #{best.number}: "
        f"objective={best.params.get('objective', base_config.objective)}, "
        f"mean Segment Recall@5={float(best.value):.5f}. "
        f"Results saved to {output_path}; database: {db_path}."
    )
    return {**payload, "results_path": str(output_path)}

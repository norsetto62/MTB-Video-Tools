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
from ..ml.lightgbm.model import LightGBMConfig
from ..ml.lightgbm.training import train_lightgbm_group_fold, train_lightgbm_lovo
from ..dataset.manifest import load_manifest
from .training import (
    _training_config,
    _section,
    prepare_lightgbm_manifest_data,
)

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "objective": {"choices": ["lambdarank", "rank_xendcg"]},
    "num_leaves": {"choices": [15, 31, 63]},
    # Each candidate is trained for this fixed number of rounds; no fold early stopping.
    "num_iterations": {"choices": [300, 500, 800]},
    "learning_rate": {"low": 0.02, "high": 0.08, "log": True},
    "max_depth": {"choices": [3, 4, 5, 6]},
    "min_data_in_leaf": {"choices": [10, 20, 50, 100]},
    "feature_fraction": {"low": 0.6, "high": 1.0},
    "bagging_fraction": {"low": 0.6, "high": 1.0},
    "lambda_l1": {"low": 1e-4, "high": 10.0, "log": True},
    "lambda_l2": {"low": 1e-4, "high": 10.0, "log": True},
    "min_split_gain": {"low": 0.0, "high": 0.2},
}


def _hpo_section(settings: Any):
    if "lightgbm_hpo" in settings:
        raise ValueError(
            "The 'lightgbm_hpo' configuration section is obsolete; use 'hpo'."
        )
    return _section(settings, "hpo")


def _validated_search_space(settings: Any) -> dict[str, dict[str, Any]]:
    section = _hpo_section(settings)
    raw = section.get("search_space", {})
    if isinstance(raw, Config):
        raw = raw.to_dict()
    if not isinstance(raw, dict):
        raise ValueError("hpo.search_space must be a mapping.")

    # User entries override defaults; omitted parameters remain tunable.
    space: dict[str, dict[str, Any]] = {
        name: dict(spec) for name, spec in DEFAULT_SEARCH_SPACE.items()
    }
    normalized_raw = dict(raw)

    allowed = set(LightGBMConfig.__dataclass_fields__) - {
        "task", "early_stopping_rounds", "ndcg_eval_at", "random_seed", "verbosity"
    }
    unknown = sorted(set(normalized_raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown LightGBM HPO search parameter(s): {unknown}.")

    integer_params = {"num_leaves", "max_depth", "min_data_in_leaf", "num_iterations", "bagging_frequency"}
    for name, value in normalized_raw.items():
        if isinstance(value, Config):
            value = value.to_dict()

        if not isinstance(value, dict):
            spec = {"fixed": value}
        else:
            extra = set(value) - {"fixed", "choices", "low", "high", "step", "log"}
            if extra:
                raise ValueError(f"Unknown search-space option(s) for {name!r}: {sorted(extra)}.")
            if "fixed" in value:
                if len(value) != 1:
                    raise ValueError(f"Search parameter {name!r}: 'fixed' cannot be combined with other options.")
                spec = {"fixed": value["fixed"]}
            elif "choices" in value:
                if set(value) != {"choices"}:
                    raise ValueError(f"Search parameter {name!r}: 'choices' cannot be combined with bounds.")
                choices = value["choices"]
                if not isinstance(choices, (tuple, list)) or not choices:
                    raise ValueError(f"Search choices for {name!r} must be a non-empty list.")
                spec = {"choices": list(choices)}
            else:
                if "low" not in value or "high" not in value:
                    raise ValueError(f"Search parameter {name!r} requires low/high, choices, or fixed.")
                if "log" in value and not isinstance(value["log"], bool):
                    raise ValueError(f"Search parameter {name!r}: log must be true or false.")
                low_raw, high_raw = value["low"], value["high"]
                if isinstance(low_raw, bool) or isinstance(high_raw, bool):
                    raise ValueError(f"Search bounds for {name!r} must be numeric.")
                low, high = float(low_raw), float(high_raw)
                if not np.isfinite([low, high]).all() or low > high:
                    raise ValueError(f"Invalid low/high bounds for {name!r}.")
                log_scale = bool(value.get("log", False))
                if log_scale and low <= 0:
                    raise ValueError(f"Log-scaled search parameter {name!r} requires low > 0.")
                spec = {"low": low, "high": high, "log": log_scale}
                if "step" in value:
                    step = float(value["step"])
                    if not np.isfinite(step) or step <= 0:
                        raise ValueError(f"Search step for {name!r} must be finite and positive.")
                    spec["step"] = step
                if name in integer_params:
                    if not low.is_integer() or not high.is_integer():
                        raise ValueError(f"Integer search parameter {name!r} requires integer low/high bounds.")
                    if "step" in spec and not float(spec["step"]).is_integer():
                        raise ValueError(f"Integer search parameter {name!r} requires an integer step.")
                    if "step" in spec:
                        spec["step"] = int(spec["step"])
                    if log_scale and spec.get("step", 1) != 1:
                        raise ValueError(
                            f"Integer search parameter {name!r} cannot combine log=true with step != 1."
                        )
                    spec["low"] = int(low)
                    spec["high"] = int(high)
                elif "step" in spec and spec["log"]:
                    raise ValueError(f"Search parameter {name!r} cannot combine log=true and step.")
        space[name] = spec

    # Validate objective choices, while allowing a single fixed objective.
    objective_spec = space["objective"]
    valid_objectives = {"lambdarank", "rank_xendcg"}
    if "fixed" in objective_spec:
        if objective_spec["fixed"] not in valid_objectives:
            raise ValueError("hpo.search_space.objective must be 'lambdarank' or 'rank_xendcg'.")
    elif "choices" in objective_spec:
        choices = objective_spec["choices"]
        if len(choices) < 2 or len(set(choices)) != len(choices) or any(
            item not in valid_objectives for item in choices
        ):
            raise ValueError(
                "hpo.search_space.objective choices must contain at least two "
                "distinct values selected from ['lambdarank', 'rank_xendcg']; "
                "use fixed for a single objective."
            )
    else:
        raise ValueError("hpo.search_space.objective must use choices or fixed.")

    return space

def _effective_params(
    base_config: LightGBMConfig,
    space: dict[str, dict[str, Any]],
    suggested_params: dict[str, Any],
) -> dict[str, Any]:
    """Return the complete LightGBM config, including fixed HPO parameters.

    Optuna's FrozenTrial.params contains only suggested parameters; fixed
    values from the search space must be applied explicitly over base_config.
    Suggested values take precedence if a parameter is sampled by Optuna.
    """
    fixed_params = {
        name: spec["fixed"] for name, spec in space.items() if "fixed" in spec
    }
    return asdict(replace(base_config, **{**fixed_params, **suggested_params}))


def _suggest(trial: optuna.Trial, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    integer_params = {"num_leaves", "max_depth", "min_data_in_leaf", "num_iterations", "bagging_frequency"}
    for name, spec in space.items():
        if "fixed" in spec:
            params[name] = spec["fixed"]
        elif "choices" in spec:
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif name in integer_params:
            params[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)),
                log=spec.get("log", False),
            )
        elif "step" in spec:
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], step=spec["step"]
            )
        else:
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
    return params

def run_hpo(manifest_path: str | Path, settings: Any) -> dict[str, Any]:
    """Tune LightGBM hyperparameters by mean per-video MTB-3 Segment Recall@5.

    Each video is a separate query for LambdaRank and for evaluation. Optuna
    maximizes the unweighted mean Segment Recall@5 across held-out LOVO groups.
    """
    section = _hpo_section(settings)
    n_trials = section.get("n_trials", 30)
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError("hpo.n_trials must be a positive integer.")
    study_name = section.get("study_name", "ranking_hpo")
    if not isinstance(study_name, str) or not study_name.strip():
        raise ValueError("hpo.study_name must be a non-empty string.")
    # Isolate fixed-iteration trials from all historical early-stopped studies.
    if not study_name.endswith("_segment_recall_at_5_fixed_iters"):
        if not study_name.endswith("_segment_recall_at_5"):
            study_name = f"{study_name}_segment_recall_at_5"
        study_name = f"{study_name}_fixed_iters"
    progress = section.get("progress", True)
    if not isinstance(progress, bool):
        raise ValueError("hpo.progress must be true or false.")

    def report(message: str) -> None:
        # HPO progress is deliberately independent of logging.level, so WARNING
        # can continue to suppress noisy cache diagnostics.
        if progress:
            print(f"[LightGBM HPO] {message}", flush=True)

    report(f"Preparing manifest data from {manifest_path}.")

    base_config = _training_config(settings)
    search_space = _validated_search_space(settings)
    report("Effective Optuna search space: " + json.dumps(search_space, sort_keys=True))

    # Proxy groups are an explicit HPO-only subset: filter before feature
    # preparation so non-proxy videos are not decoded or extracted.
    proxy_value = section.get("proxy_test_groups")
    selected_records = None
    if proxy_value is not None:
        if isinstance(proxy_value, Config):
            proxy_value = proxy_value.to_dict()
        if not isinstance(proxy_value, (tuple, list)) or not proxy_value:
            raise ValueError("hpo.proxy_test_groups must be a non-empty list of group IDs.")
        proxy_groups: list[str] = []
        for item in proxy_value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("hpo.proxy_test_groups must contain non-empty string group IDs.")
            group = item.strip()
            if group in proxy_groups:
                raise ValueError(f"hpo.proxy_test_groups contains duplicate group ID {group!r}.")
            proxy_groups.append(group)
        report(f"Loading manifest to select proxy groups: {', '.join(proxy_groups)}.")
        all_records = load_manifest(manifest_path, verify_files=True)
        available_groups = {record.group_id for record in all_records}
        unknown_groups = sorted(set(proxy_groups) - available_groups)
        if unknown_groups:
            raise ValueError(
                "hpo.proxy_test_groups not found in manifest: "
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
    db_path = Path(section.get("storage_path", "data/db/hpo.db")).expanduser()
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

    objective_description = (
        search_space["objective"].get("choices", search_space["objective"].get("fixed"))
    )
    logger.info(
        "Starting HPO study %r: %d new trials, objectives=%s, groups=%s",
        study_name, n_trials, objective_description,
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
    fixed_search_params = {
        name: spec["fixed"] for name, spec in search_space.items() if "fixed" in spec
    }

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
        "fixed_search_params": fixed_search_params,
        "best_params": _effective_params(base_config, search_space, best.params),
        "best_fold_segment_recall_at_5": best.user_attrs.get("fold_segment_recall_at_5", {}),
        "trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "effective_params": _effective_params(base_config, search_space, trial.params),
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


def run_nested_lightgbm_evaluation(
    manifest_path: str | Path, settings: Any
) -> dict[str, Any]:
    """Estimate the complete HPO+fit procedure with outer leave-one-group-out CV.

    Features are prepared once. For each outer group, Optuna tunes parameters
    using only inner LOVO folds drawn from the remaining groups. The selected
    configuration is then refit on all outer-training groups for its fixed
    num_iterations and scored exactly once on the untouched outer group.
    Each outer fold gets a fresh in-memory study; no trials or labels cross
    between outer folds.
    """
    section = _hpo_section(settings)
    n_trials = section.get("n_trials", 30)
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError("hpo.n_trials must be a positive integer.")
    progress = section.get("progress", True)
    if not isinstance(progress, bool):
        raise ValueError("hpo.progress must be true or false.")

    def report(message: str) -> None:
        if progress:
            print(f"[LightGBM nested CV] {message}", flush=True)

    base_config = _training_config(settings)
    search_space = _validated_search_space(settings)
    report(f"Preparing all manifest data once from {manifest_path}.")
    prepared, group_by_video, data_config = prepare_lightgbm_manifest_data(
        manifest_path, settings, progress_callback=report
    )
    groups = sorted(set(group_by_video.values()))
    if len(groups) < 3:
        raise ValueError(
            "Nested group CV requires at least three annotated groups: "
            "one outer test group and at least two groups for inner LOVO."
        )

    outer_results: list[dict[str, Any]] = []
    all_outer_video_recalls: list[float] = []
    for outer_index, outer_group in enumerate(groups):
        inner_groups = [group for group in groups if group != outer_group]
        inner_video_ids = sorted(
            video_id for video_id in prepared
            if group_by_video[video_id] in inner_groups
        )
        inner_data = {video_id: prepared[video_id] for video_id in inner_video_ids}
        inner_group_map = {video_id: group_by_video[video_id] for video_id in inner_video_ids}
        report(
            f"Outer fold {outer_index + 1}/{len(groups)}: test={outer_group}; "
            f"inner groups={inner_groups}; tuning {n_trials} trials."
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=base_config.random_seed + outer_index
            ),
        )

        def objective(trial: optuna.Trial) -> float:
            params = _suggest(trial, search_space)
            trial_config = replace(base_config, **params)
            inner_folds = train_lightgbm_lovo(
                inner_data,
                inner_group_map,
                config=trial_config,
                data_config={
                    **data_config,
                    "nested_cv_role": "inner_validation",
                    "outer_test_group_excluded": outer_group,
                    "outer_fold": outer_index,
                    "trial": trial.number,
                },
            )
            recalls: list[float] = []
            per_fold: dict[str, float | None] = {}
            for inner_fold in inner_folds:
                metrics = inner_fold.summary.get("validation_metrics", {})
                per_video = metrics.get("per_video", {})
                values = [
                    float(item["recall"])
                    for item in per_video.values()
                    if item.get("recall") is not None
                    and np.isfinite(float(item["recall"]))
                ]
                per_fold[inner_fold.held_out_group] = (
                    float(np.mean(values)) if values else None
                )
                recalls.extend(values)
            if not recalls:
                raise ValueError(
                    f"Outer group {outer_group!r}, inner trial {trial.number} "
                    "has no inner-validation videos with class-3 events."
                )
            score = float(np.mean(recalls))
            trial.set_user_attr("inner_fold_segment_recall_at_5", per_fold)
            trial.set_user_attr("inner_evaluated_video_count", len(recalls))
            return score

        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
        completed = [
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
        ]
        if not completed:
            raise RuntimeError(
                f"Nested CV outer fold {outer_group!r} produced no completed HPO trials."
            )
        best = study.best_trial
        best_params = _effective_params(base_config, search_space, best.params)
        selected_config = replace(base_config, **best_params)

        _, outer_fold = train_lightgbm_group_fold(
            prepared,
            group_by_video,
            held_out_group=outer_group,
            config=selected_config,
            data_config={
                **data_config,
                "nested_cv_role": "outer_test",
                "outer_fold": outer_index,
                "inner_hpo_best_value": float(best.value),
                "inner_hpo_best_params": best_params,
            },
        )
        outer_metrics = outer_fold.summary.get("validation_metrics", {})
        outer_per_video = outer_metrics.get("per_video", {})
        # segment_recall_at_k numbers videos by their position in the
        # concatenated validation arrays; translate those positions back to
        # stable manifest video IDs for an interpretable report.
        outer_recalls = {}
        for video_index, video_id in enumerate(outer_fold.validation_video_ids):
            item = outer_per_video.get(str(video_index))
            if (
                item is not None
                and item.get("recall") is not None
                and np.isfinite(float(item["recall"]))
            ):
                outer_recalls[video_id] = float(item["recall"])
        all_outer_video_recalls.extend(outer_recalls.values())
        outer_score = (
            float(np.mean(list(outer_recalls.values()))) if outer_recalls else None
        )
        outer_results.append({
            "outer_group": outer_group,
            "inner_groups": inner_groups,
            "inner_best_trial_number": best.number,
            "inner_best_value": float(best.value),
            "inner_best_params": best_params,
            "inner_best_fold_scores": best.user_attrs.get(
                "inner_fold_segment_recall_at_5", {}
            ),
            "outer_segment_recall_at_5": outer_score,
            "outer_per_video_segment_recall_at_5": outer_recalls,
            "outer_events": outer_metrics.get("events", 0),
            "outer_evaluated_videos": len(outer_recalls),
            "outer_training_video_ids": list(outer_fold.train_video_ids),
            "outer_test_video_ids": list(outer_fold.validation_video_ids),
            "outer_evaluation_history": outer_fold.summary.get(
                "evaluation_history", {}
            ),
        })
        report(
            f"Outer fold {outer_index + 1}/{len(groups)} complete: "
            f"group={outer_group}, inner best={float(best.value):.5f}, "
            f"outer recall@5={outer_score if outer_score is not None else 'undefined'}."
        )

    outer_fold_scores = [
        float(item["outer_segment_recall_at_5"])
        for item in outer_results
        if item["outer_segment_recall_at_5"] is not None
    ]
    mean_score = (
        float(np.mean(all_outer_video_recalls))
        if all_outer_video_recalls else None
    )
    std_score = (
        float(np.std(all_outer_video_recalls, ddof=1))
        if len(all_outer_video_recalls) > 1 else (0.0 if all_outer_video_recalls else None)
    )
    mean_fold_score = (
        float(np.mean(outer_fold_scores)) if outer_fold_scores else None
    )
    std_fold_score = (
        float(np.std(outer_fold_scores, ddof=1))
        if len(outer_fold_scores) > 1
        else (0.0 if outer_fold_scores else None)
    )
    output_dir = Path(section.get("output_dir", "output/hpo")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lightgbm_nested_cv_segment_recall_at_5.json"
    payload = {
        "evaluation": "nested_leave_one_group_out_hpo",
        "metric": "segment_recall_at_5",
        "aggregation": "unweighted_mean_over_outer_test_videos_with_class3_events",
        "direction": "maximize",
        "manifest": str(Path(manifest_path).expanduser().resolve()),
        "groups": groups,
        "outer_fold_count": len(groups),
        "inner_trials_per_outer_fold": n_trials,
        "prepared_video_count": len(prepared),
        "evaluated_outer_video_count": len(all_outer_video_recalls),
        "mean_outer_segment_recall_at_5": mean_score,
        "std_outer_video_segment_recall_at_5": std_score,
        "mean_outer_group_segment_recall_at_5": mean_fold_score,
        "std_outer_group_segment_recall_at_5": std_fold_score,
        "outer_folds": outer_results,
        "interpretation": (
            "Estimates the HPO-and-refit procedure. It is not the score of a "
            "single final model trained on all groups."
        ),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["results_path"] = str(output_path)
    report(
        f"Nested CV complete: mean outer per-video Segment Recall@5="
        f"{mean_score if mean_score is not None else 'undefined'}; "
        f"results saved to {output_path}."
    )
    return payload

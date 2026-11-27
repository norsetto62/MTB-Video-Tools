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
from ..ml.ranking_metrics import METRIC_ID
from ..ml.lightgbm.training import train_lightgbm_group_fold, train_lightgbm_lovo
from .training import (
    _training_config,
    _section,
    prepare_lightgbm_manifest_data,
)

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "objective": {"choices": ["lambdarank", "rank_xendcg"]},
    "num_leaves": {"low": 15, "high": 63},
    # Each candidate is trained for this fixed number of rounds; no fold early stopping.
    "num_iterations": {"low": 300, "high": 800, "step": 100},
    "learning_rate": {"low": 0.02, "high": 0.08, "log": True},
    # Keep max_depth at the training-config default (6) while tuning leaves;
    # jointly sampling both creates many redundant/capped combinations.
    "min_data_in_leaf": {"low": 10, "high": 100, "log": True},
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

def run_hpo(
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
            print(f"[LightGBM HPO] {message}", flush=True)

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
    outer_total_events = outer_total_hits = 0
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
                collect_evaluation_history=False,
                data_config={
                    **data_config,
                    "nested_cv_role": "inner_validation",
                    "outer_test_group_excluded": outer_group,
                    "outer_fold": outer_index,
                    "trial": trial.number,
                },
            )
            total_events = total_hits = 0
            per_fold: dict[str, float | None] = {}
            for inner_fold in inner_folds:
                metrics = inner_fold.summary.get("validation_metrics", {})
                events = int(metrics.get("events", 0))
                hits = int(metrics.get("hit_events", 0))
                per_fold[inner_fold.held_out_group] = hits / events if events else None
                total_events += events
                total_hits += hits
            if total_events == 0:
                raise ValueError(
                    f"Outer group {outer_group!r}, inner trial {trial.number} "
                    "has no inner-validation class-3 events."
                )
            score = float(total_hits / total_events)
            trial.set_user_attr("inner_fold_candidate_event_recall_at_10_v2", per_fold)
            trial.set_user_attr("inner_event_count", total_events)
            trial.set_user_attr("inner_event_hits", total_hits)
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
            collect_evaluation_history=True,
            data_config={
                **data_config,
                "nested_cv_role": "outer_test",
                "outer_fold": outer_index,
                "inner_hpo_best_value": float(best.value),
                "inner_hpo_best_params": best_params,
            },
        )
        history = outer_fold.summary.get("evaluation_history", {})
        recall_curve = history.get("validation", {}).get(METRIC_ID, [])
        if recall_curve:
            peak_offset = int(np.argmax(recall_curve))
            iteration_diagnostic = {
                "configured_iterations": int(selected_config.num_iterations),
                "peak_iteration": peak_offset + 1,
                "peak_candidate_event_recall": float(recall_curve[peak_offset]),
                "candidate_recall_at_configured_iteration": float(recall_curve[-1]),
                "peak_minus_final": float(recall_curve[peak_offset] - recall_curve[-1]),
                "interpretation": "post-hoc diagnostic only; never used for iteration or hyperparameter selection",
            }
        else:
            iteration_diagnostic = {
                "configured_iterations": int(selected_config.num_iterations),
                "curve_available": False,
            }
        outer_metrics = outer_fold.summary.get("validation_metrics", {})
        outer_per_video = outer_metrics.get("per_video", {})
        # candidate event metrics number videos by their position in the
        # concatenated validation arrays; translate those positions back to
        # stable manifest video IDs for an interpretable report.
        outer_recalls = {}
        for video_index, video_id in enumerate(outer_fold.validation_video_ids):
            item = outer_per_video.get(str(video_index))
            if (
                item is not None
                and item.get("event_recall") is not None
                and np.isfinite(float(item["event_recall"]))
            ):
                outer_recalls[video_id] = float(item["event_recall"])
        all_outer_video_recalls.extend(outer_recalls.values())
        outer_events = int(outer_metrics.get("events", 0))
        outer_hits = int(outer_metrics.get("hit_events", 0))
        outer_total_events += outer_events
        outer_total_hits += outer_hits
        outer_score = float(outer_hits / outer_events) if outer_events else None
        outer_results.append({
            "outer_group": outer_group,
            "inner_groups": inner_groups,
            "inner_best_trial_number": best.number,
            "inner_best_value": float(best.value),
            "inner_best_params": best_params,
            "inner_best_fold_scores": best.user_attrs.get(
                "inner_fold_candidate_event_recall_at_10_v2", {}
            ),
            "outer_candidate_event_recall_at_10_v2": outer_score,
            "outer_per_video_candidate_event_recall_at_10_v2": outer_recalls,
            "outer_events": outer_metrics.get("events", 0),
            "outer_evaluated_videos": len(outer_recalls),
            "outer_training_video_ids": list(outer_fold.train_video_ids),
            "outer_test_video_ids": list(outer_fold.validation_video_ids),
            "outer_evaluation_history": history,
            "iteration_curve_diagnostic": iteration_diagnostic,
        })
        report(
            f"Outer fold {outer_index + 1}/{len(groups)} complete: "
            f"group={outer_group}, inner best={float(best.value):.5f}, "
            f"outer candidate event recall={outer_score if outer_score is not None else 'undefined'}."
        )

    normalization_name = data_config["feature_normalization"]
    normalization_slug = str(normalization_name).replace("per_video_", "")

    # The nested outer-fold studies above remain temporary and isolated.
    # The final full-dataset LOVO study is the persistent, resumable study used
    # for production parameter selection and interactive fine-tuning.
    configured_study_name = section.get("study_name", "ranking_hpo")
    if not isinstance(configured_study_name, str) or not configured_study_name.strip():
        raise ValueError("hpo.study_name must be a non-empty string.")
    # Keep Optuna histories for the old window-level NMS metric isolated.
    study_name = f"{configured_study_name.strip()}_{METRIC_ID}"

    # Preserve the old HPO proxy-group option for the production-parameter
    # search only. Nested evaluation above always uses every manifest group.
    final_prepared = prepared
    final_group_by_video = group_by_video
    proxy_value = section.get("proxy_test_groups")
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
        unknown_groups = sorted(set(proxy_groups) - set(groups))
        if unknown_groups:
            raise ValueError(
                "hpo.proxy_test_groups not found in manifest: " + ", ".join(unknown_groups)
            )
        final_video_ids = sorted(
            video_id for video_id, group in group_by_video.items()
            if group in proxy_groups
        )
        final_prepared = {video_id: prepared[video_id] for video_id in final_video_ids}
        final_group_by_video = {
            video_id: group_by_video[video_id] for video_id in final_video_ids
        }
        if len(set(final_group_by_video.values())) < 2:
            raise ValueError("Proxy-group HPO requires at least two groups for LOVO evaluation.")
        report(
            "Final parameter-selection proxy groups: "
            + ", ".join(proxy_groups)
            + f" ({len(final_prepared)} videos)."
        )

    db_path = Path(section.get("storage_path", "data/db/hpo_study.db")).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_path.resolve().as_posix()}"
    final_study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=base_config.random_seed + len(groups)),
    )
    report(
        f"Nested evaluation complete. Persistent study '{study_name}' opened at {db_path}; "
        f"running {n_trials} additional final HPO trials across "
        f"{len(set(final_group_by_video.values()))} groups "
        f"({len(final_study.trials)} existing trial records)."
    )

    def final_objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, search_space)
        trial_config = replace(base_config, **params)
        folds = train_lightgbm_lovo(
            final_prepared,
            final_group_by_video,
            config=trial_config,
            collect_evaluation_history=False,
            data_config={
                **data_config,
                "hpo_role": "final_full_dataset_parameter_selection",
                "manifest": str(Path(manifest_path).expanduser().resolve()),
                "ranking_hpo_trial": trial.number,
            },
        )
        total_events = total_hits = 0
        per_fold: dict[str, float | None] = {}
        for fold in folds:
            metrics = fold.summary.get("validation_metrics", {})
            events = int(metrics.get("events", 0))
            hits = int(metrics.get("hit_events", 0))
            per_fold[fold.held_out_group] = hits / events if events else None
            total_events += events
            total_hits += hits
        if total_events == 0:
            raise ValueError(
                f"Final HPO trial {trial.number} produced no validation class-3 events."
            )
        score = float(total_hits / total_events)
        trial.set_user_attr("metric", "candidate_event_recall_at_10_v2")
        trial.set_user_attr("pooled_candidate_event_recall_at_10_v2", score)
        trial.set_user_attr("fold_candidate_event_recall_at_10_v2", per_fold)
        trial.set_user_attr("total_events", total_events)
        trial.set_user_attr("total_hits", total_hits)
        trial.set_user_attr("evaluated_event_count", total_events)
        trial.set_user_attr("evaluated_folds", sorted(per_fold))
        logger.info(
            "LightGBM final HPO trial %d: objective=%s, pooled candidate event recall=%.5f, folds=%s",
            trial.number, params.get("objective", base_config.objective), score, per_fold,
        )
        return score

    final_study.optimize(final_objective, n_trials=n_trials, gc_after_trial=True)
    final_completed = [
        trial for trial in final_study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not final_completed:
        raise RuntimeError("Final full-dataset HPO produced no completed trials.")
    final_best = final_study.best_trial
    final_best_params = _effective_params(base_config, search_space, final_best.params)
    final_parameter_selection = {
        "method": "leave_one_group_out",
        "purpose": "production_parameter_selection_only",
        "study_name": study_name,
        "storage_path": str(db_path.resolve()),
        "proxy_test_groups": sorted(set(final_group_by_video.values())) if proxy_value is not None else None,
        "best_trial_number": final_best.number,
        "best_inner_pooled_candidate_event_recall_at_10_v2": float(final_best.value),
        "best_params": final_best_params,
        "best_fold_candidate_event_recall_at_10_v2": final_best.user_attrs.get(
            "fold_candidate_event_recall_at_10_v2", {}
        ),
        "total_events": final_best.user_attrs.get("total_events"),
        "total_hits": final_best.user_attrs.get("total_hits"),
        "completed_trials": len(final_completed),
        "new_trials_this_run": n_trials,
        "trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "effective_params": _effective_params(base_config, search_space, trial.params),
                "fold_candidate_event_recall_at_10_v2": trial.user_attrs.get("fold_candidate_event_recall_at_10_v2", {}),
            }
            for trial in final_completed
        ],
        "warning": (
            "This full-dataset LOVO score is used to select production parameters; "
            "it is not an unbiased estimate. Use mean_outer_candidate_event_recall_at_10_v2 "
            "from nested_cv for generalization performance."
        ),
    }
    report(
        f"Final production parameters selected: trial #{final_best.number}, "
        f"full-data LOVO pooled candidate event recall={float(final_best.value):.5f}. "
        f"Study '{study_name}' now contains {len(final_study.trials)} trial records; "
        f"database: {db_path}."
    )

    outer_fold_scores = [
        float(item["outer_candidate_event_recall_at_10_v2"])
        for item in outer_results
        if item["outer_candidate_event_recall_at_10_v2"] is not None
    ]
    mean_score = float(outer_total_hits / outer_total_events) if outer_total_events else None
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
    normalization_slug = str(normalization_name).replace("per_video_", "")
    output_path = output_dir / f"lightgbm_hpo_nested_{METRIC_ID}_{normalization_slug}.json"
    payload = {
        "evaluation": "nested_leave_one_group_out_hpo",
        "metric": METRIC_ID,
        "metric_version": METRIC_ID,
        "aggregation": "unique_class3_annotation_events_covered_by_selected_clips_at_least_1s_divided_by_total_events",
        "direction": "maximize",
        "manifest": str(Path(manifest_path).expanduser().resolve()),
        "groups": groups,
        "outer_fold_count": len(groups),
        "inner_trials_per_outer_fold": n_trials,
        "prepared_video_count": len(prepared),
        "class3_event_target_diagnostics": data_config.get("class3_event_target_diagnostics", {}),
        "evaluated_outer_video_count": len(all_outer_video_recalls),
        "outer_total_events": outer_total_events,
        "outer_total_hits": outer_total_hits,
        "mean_outer_candidate_event_recall_at_10_v2": mean_score,
        "std_outer_video_candidate_event_recall_at_10_v2": std_score,
        "mean_outer_group_candidate_event_recall_at_10_v2": mean_fold_score,
        "std_outer_group_candidate_event_recall_at_10_v2": std_fold_score,
        "outer_folds": outer_results,
        # Top-level best_params is intentionally compatible with
        # training.run_lightgbm_train_from_manifest's hpo.best_params_path.
        "best_params": final_best_params,
        "study_name": study_name,
        "study_storage_path": str(db_path.resolve()),
        "final_parameter_selection": final_parameter_selection,
        "interpretation": (
            "Estimates the HPO-and-refit procedure. It is not the score of a "
            "single final model trained on all groups. The top-level best_params "
            "comes from the separate full-dataset LOVO parameter-selection stage; "
            "its score is not an unbiased performance estimate."
        ),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["results_path"] = str(output_path)
    report(
        f"Nested CV and final parameter selection complete: pooled outer NMS Segment Recall@5="
        f"{mean_score if mean_score is not None else 'undefined'}; "
        f"results saved to {output_path}."
    )
    return payload

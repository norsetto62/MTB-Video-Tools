import json
import logging
from collections.abc import Mapping
from pathlib import Path
import numpy as np
import optuna
from datetime import datetime
from typing import Any

from ..config_loader import Config
from .lovo import run_lovo
from ..dataset.manifest import load_manifest
from .training import _combine_records, _section, _training_config
from ..ml.training import train_fixed_epochs

logger = logging.getLogger(__name__)


def run_hpo(
    manifest_path: str | Path,
    settings: Mapping[str, Any],
) -> None:
    """
    Orchestrates the Optuna HPO study, logs outputs to screen, 
    and exports winning parameters to output/hpo/.
    """
    
    if not isinstance(settings, Mapping):
        raise ValueError("HPO settings must be a configuration mapping.")
    hpo = _section(settings, "hpo")
    data = _section(settings, "data")
    paths = _section(settings, "paths")
    n_trials = hpo.get("n_trials", 0)
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError("hpo.n_trials must be a positive integer.")
    proxy_groups = tuple(hpo.get("proxy_test_groups", ()))
    if not proxy_groups or any(not isinstance(g, str) or not g.strip() for g in proxy_groups):
        raise ValueError("hpo.proxy_test_groups must contain non-empty group IDs.")
    study_name = hpo.get("study_name")
    if not isinstance(study_name, str) or not study_name.strip():
        raise ValueError("hpo.study_name must be a non-empty string.")
    training_config = _training_config(settings)
    base_threshold = float(_section(settings, "training").get(
        "positive_threshold", _section(settings, "training").get("c2_threshold", 0.50)
    ))
    train_groups = tuple(data.get("train_groups", ()))
    search_space = hpo.get("search_space", {})
    if isinstance(search_space, Config):
        search_space = search_space.to_dict()

    # Database path explicitly placed in data/db/
    db_dir = Path("data/db")
    db_dir.mkdir(parents=True, exist_ok=True)
    storage_db = f"sqlite:///{db_dir / 'hpo_study_pu.db'}"

    logger.info("Initializing Optuna study '%s' with storage: %s", study_name, storage_db)
    logger.info("Proxy validation folds assigned for tuning: %s", proxy_groups)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_db,
        load_if_exists=True,
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.TPESampler(seed=training_config.seed),
    )

    if not search_space:
        raise ValueError(
            "HPO search space is empty. Configure hpo.search_space in the training YAML."
        )
    obsolete = sorted(
        name for name in search_space
        if name.startswith("class_weight_") or name == "good_threshold"
    )
    if obsolete:
        raise ValueError(
            "PU HPO does not use supervised class weights or binary good_threshold; "
            "replace these search-space entries with positive_prior and c2_threshold. "
            "Obsolete entries: " + ", ".join(obsolete)
        )

    def suggest_parameters(trial: optuna.Trial) -> dict[str, object]:
        """Build Optuna suggestions from the validated YAML distributions."""
        params: dict[str, object] = {}
        for name, spec in search_space.items():
            if "fixed" in spec:
                params[name] = spec["fixed"]
                continue
            if "choices" in spec:
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                step = spec.get("step")
                if step is not None:
                    #step and log cannot be used together
                    params[name] = trial.suggest_float(
                        name, float(spec["low"]), float(spec["high"]),
                        step=float(step)
                    )
                else:
                    params[name] = trial.suggest_float(
                        name, float(spec["low"]), float(spec["high"]),
                        log=bool(spec.get("log", False)),
                    )
        return params

    # Define objective function closure
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        params = suggest_parameters(trial)
        lr = float(params["lr"])
        weight_decay = float(params["weight_decay"])
        dropout = float(params["dropout"])
        hidden_dim = int(params["hidden_dim"])
        positive_prior = float(params.get("positive_prior", training_config.positive_prior))
        positive_threshold = float(params.get("positive_threshold", params.get("c2_threshold", base_threshold)))

        filtered_train_groups = tuple(g for g in train_groups if g not in proxy_groups)

        trial_settings = Config(settings).with_overrides({
            "training.lr": lr,
            "training.weight_decay": weight_decay,
            "training.dropout": dropout,
            "training.hidden_dim": hidden_dim,
            "training.positive_prior": positive_prior,
            "training.positive_threshold": positive_threshold,
            "data.train_groups": filtered_train_groups,
            "data.validation_groups": tuple(proxy_groups),
        })

        # Explicitly restrict evaluation to proxy folds and the configured non-proxy
        # training pool. run_lovo validates these group IDs against the manifest.
        lovo_results = run_lovo(
            manifest_path=manifest_path,
            settings=trial_settings,
            test_groups=tuple(proxy_groups),
            train_groups=tuple(filtered_train_groups),
        )

        # U is unlabeled, not a negative class. Reward recall on P while
        # penalizing how often U is selected as positive; do not compute F1
        # or precision from pseudo-negatives.
        folds = lovo_results.get("folds", [])
        recalls = [f["metrics"]["positive_recall_on_P"] for f in folds
                   if f.get("metrics", {}).get("positive_recall_on_P") is not None]
        u_rates = [f["metrics"]["selected_fraction_of_U"] for f in folds
                   if f.get("metrics", {}).get("selected_fraction_of_U") is not None]
        recall = float(np.mean(recalls)) if recalls else 0.0
        u_selection = float(np.mean(u_rates)) if u_rates else 1.0
        trial.set_user_attr("positive_recall_on_P", recall)
        trial.set_user_attr("selected_fraction_of_U", u_selection)
        trial.set_user_attr("positive_prior", positive_prior)
        trial.set_user_attr("positive_threshold", positive_threshold)
        return recall, u_selection

    # Run Optuna optimization loop

    # Run the requested number of new trials in this execution session
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    
    if n_trials > 0:
        logger.info(
            "Running %d new trials for study (existing completed trials: %d)...",
            n_trials,
            completed_trials,
        )
        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    else:
        logger.info("n_trials is set to 0. Skipping optimization.")

    # Multi-objective studies have a Pareto frontier, not a single best_trial.
    pareto_trials = sorted(
        study.best_trials,
        key=lambda trial: (
            trial.user_attrs.get("selected_fraction_of_U", 1.0),
            -trial.user_attrs.get("positive_recall_on_P", 0.0),
            trial.number,
        ),
    )
    if not pareto_trials:
        raise RuntimeError("HPO produced no completed Pareto-efficient trials.")

    # Production refit needs one explicit choice. Select the Pareto trial with
    # highest P recall; break ties by lower U selection. This is a post-study
    # deployment policy, not a combined optimization objective.
    production_trial = min(
        pareto_trials,
        key=lambda trial: (
            -trial.user_attrs.get("positive_recall_on_P", 0.0),
            trial.user_attrs.get("selected_fraction_of_U", 1.0),
            trial.number,
        ),
    )
    production_attrs = production_trial.user_attrs
    pareto_payload = [
        {
            "trial_number": trial.number,
            "objectives": {
                "positive_recall_on_P": trial.values[0],
                "selected_fraction_of_U": trial.values[1],
            },
            "params": trial.params,
            "metrics": {
                "positive_recall_on_P": trial.user_attrs.get("positive_recall_on_P", 0.0),
                "selected_fraction_of_U": trial.user_attrs.get("selected_fraction_of_U", 1.0),
                "positive_prior": trial.user_attrs.get("positive_prior", training_config.positive_prior),
                "positive_threshold": trial.user_attrs.get("positive_threshold", base_threshold),
            },
        }
        for trial in pareto_trials
    ]
    output_payload = {
        "study_name": study_name,
        "objectives": [
            {"name": "positive_recall_on_P", "direction": "maximize"},
            {"name": "selected_fraction_of_U", "direction": "minimize"},
        ],
        "pareto_front": pareto_payload,
        "production_selection_policy": "highest P recall on Pareto front; ties broken by lower U selection",
        "production_trial_number": production_trial.number,
        "production_params": production_trial.params,
        "production_metrics": {
            "positive_recall_on_P": production_attrs.get("positive_recall_on_P", 0.0),
            "selected_fraction_of_U": production_attrs.get("selected_fraction_of_U", 1.0),
            "positive_prior": production_attrs.get("positive_prior", training_config.positive_prior),
            "positive_threshold": production_attrs.get("positive_threshold", base_threshold),
        },
    }

    # Mirror results to screen, including the non-dominated trade-off set.
    logger.info("=" * 60)
    logger.info("MULTI-OBJECTIVE HPO COMPLETE")
    logger.info("Pareto-efficient trials: %d", len(pareto_trials))
    for item in pareto_payload:
        metrics = item["metrics"]
        logger.info(
            "Trial %d -> P recall: %.4f | U selected: %.4f",
            item["trial_number"],
            metrics["positive_recall_on_P"],
            metrics["selected_fraction_of_U"],
        )
    logger.info(
        "Production trial %d selected by highest P recall (tie-break: lower U selection).",
        production_trial.number,
    )
    logger.info("=" * 60)

    # Save output mirror json to output/hpo/
    output_dir = Path("output/hpo")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{study_name}_pareto_results.json"
    
    with open(json_path, "w") as f:
        json.dump(output_payload, f, indent=4)
        
    logger.info("Successfully exported winning configuration JSON to %s", json_path)

    # Execute Final Production Master Refit using the explicit Pareto-front selection policy
    log_msg = (
            f"Starting Final Production Master Refit with selected Pareto trial {production_trial.number}..."
    )

    # Let the logger do its job if INFO is enabled
    logger.info(log_msg)

    # If the logger is silenced at WARNING or above, print directly to the console
    if logger.getEffectiveLevel() >= logging.WARNING:
            print(log_msg)

    resolved_params = dict(production_trial.params)
    for name, spec in search_space.items():
        if "fixed" in spec:
            resolved_params[name] = spec["fixed"]

    final_settings = Config(settings).with_overrides({
        "training.lr": resolved_params["lr"],
        "training.weight_decay": resolved_params["weight_decay"],
        "training.dropout": resolved_params["dropout"],
        "training.positive_prior": resolved_params.get("positive_prior", training_config.positive_prior),
        "training.positive_threshold": resolved_params.get(
            "positive_threshold", resolved_params.get("c2_threshold", base_threshold)
        ),
        "training.hidden_dim": resolved_params["hidden_dim"],
    })
    final_training = _training_config(final_settings)

    result = run_lovo(manifest_path, final_settings)
    # Determine the optimal epoch count for the final master refit 
    # (e.g., taking the mean or median of the consensus selected epochs from each fold)
    fold_epochs = [
        fold.get("consensus_selected_epochs") or fold.get("selected_epochs") 
        for fold in result["folds"]
    ]
    # Fallback to training_config.epochs if epochs aren't explicitly tracked per fold
    mean_epochs = int(np.mean([e for e in fold_epochs if e is not None])) if fold_epochs else training_config.epochs

    print(f"Refitting master production model on all groups for {mean_epochs} epochs...")

    # Load and combine all records using the codebase's native helper
    records = load_manifest(manifest_path, verify_files=True)
    all_windows, all_targets = _combine_records(records, final_settings)

    # Define your production checkpoint destination
    experiment = _section(settings, "experiment")
    base_exp_name = experiment.get("name", "baseline")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_exp_name = f"{base_exp_name}_{timestamp}"
    models_dir = Path(paths.get("models_dir", "models")).expanduser() / "experiments"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / f"{unique_exp_name}_master_production.pt"

    # Train and save using your exact fixed-epoch function
    model, scaler, train_summary = train_fixed_epochs(
        train_windows=all_windows,
        train_targets=all_targets,
        config=final_training,
        epochs=mean_epochs,
        checkpoint_path=checkpoint_path,
        data_config=result.get("data_config"),
        metadata={"experiment_name": unique_exp_name},
    )

    print(f"Master production model successfully saved to {checkpoint_path}")
    logger.info("Final production model checkpoint successfully saved.")
 
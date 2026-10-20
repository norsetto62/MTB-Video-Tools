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
    train_groups = tuple(data.get("train_groups", ()))
    search_space = hpo.get("search_space", {})
    if isinstance(search_space, Config):
        search_space = search_space.to_dict()

    # Database path explicitly placed in data/db/
    db_dir = Path("data/db")
    db_dir.mkdir(parents=True, exist_ok=True)
    storage_db = f"sqlite:///{db_dir / 'hpo_study.db'}"

    logger.info("Initializing Optuna study '%s' with storage: %s", study_name, storage_db)
    logger.info("Proxy validation folds assigned for tuning: %s", proxy_groups)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_db,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=training_config.seed),
    )

    if not search_space:
        raise ValueError(
            "HPO search space is empty. Configure hpo.search_space in the training YAML."
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
    def objective(trial: optuna.Trial) -> float:
        params = suggest_parameters(trial)
        lr = float(params["lr"])
        weight_decay = float(params["weight_decay"])
        dropout = float(params["dropout"])
        class_weights = [float(params["class_weight_0"]), 1.0, float(params["class_weight_2"])]
        hidden_dim = int(params["hidden_dim"])

        # --- ADD THRESHOLD HYPERPARAMETER ---
        # Forces Optuna to evaluate configurations under different safety margins
        c2_threshold = float(params["c2_threshold"])

        filtered_train_groups = tuple(g for g in train_groups if g not in proxy_groups)

                trial_settings = Config(settings).with_overrides({
            "training.lr": lr,
            "training.weight_decay": weight_decay,
            "training.dropout": dropout,
            "training.class_weights": tuple(class_weights),
            "training.hidden_dim": hidden_dim,
            "training.c2_threshold": c2_threshold,
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

        # Aggregate confusion matrices across validation folds
        folds = lovo_results.get("folds", [])
        total_cm = np.zeros((3, 3), dtype=np.float64)

        all_fold_class_f1s = []
        for f in folds:
            metrics = f.get("metrics", {})
            cm = metrics.get("confusion_matrix")
            if cm is not None:
                total_cm += np.array(cm, dtype=np.float64)

            class_f1_dict = metrics.get("class_f1", {})
            if not class_f1_dict:
                values = metrics.get("per_class_f1")
                if isinstance(values, (list, tuple)) and len(values) >= 3:
                    class_f1_dict = {str(i): values[i] for i in range(3)}
            if class_f1_dict:
                arr = np.array([class_f1_dict.get("0", 0.0), class_f1_dict.get("1", 0.0), class_f1_dict.get("2", 0.0)])
                all_fold_class_f1s.append(arr)

        if all_fold_class_f1s:
            mean_per_class_f1 = np.mean(all_fold_class_f1s, axis=0)
            c0_f1, c1_f1, c2_f1 = float(mean_per_class_f1[0]), float(mean_per_class_f1[1]), float(mean_per_class_f1[2])
        else:
            c0_f1, c1_f1, c2_f1 = 0.0, 0.0, 0.0

        # --- CORRECTED RATE-BASED ASYMMETRIC OBJECTIVE ---
        total_pred=np.sum(total_cm, axis=0, keepdims=True)
        total_true=np.sum(total_cm, axis=1, keepdims=True)
        empty_pred = (total_pred.squeeze() == 0)
        empty_true = (total_true.squeeze() == 0)
        safe_pred = np.where(total_pred == 0, 1.0, total_pred)
        safe_true = np.where(total_true == 0, 1.0, total_true)
        recall = np.diag(total_cm/safe_true, k=0).copy()
        precision = np.diag(total_cm/safe_pred, k=0).copy()
        recall[empty_true] = 1.0
        precision[empty_pred] = 1.0
        #f1 = 2.0 * recall * precision / (recall + precision)
        f2 = 5.0 * recall * precision / (recall + 4.0 * precision)
        f05 = 1.25 * recall * precision / (recall + 0.25 * precision)

        # Multiplicative Scaling (harmonic mean of class_0 f2, class_1 f2 and class_2 f0.5)
        den = f2[1] * f05[2] + f2[0] * f05[2] + f2[0] * f2[1]
        if den == 0.0:
            score = 0.0
        else:
            score = 3.0 * f2[0] * f2[1] * f05[2] / den
        """
        Since f05[2] is so importanto to us, we could use weights:

        score = (1+a*a+b*b)*(c0_f2 * c1_f2 * c2_f05)/(a*a*c1_f2*c2_f05+b*b*c0_f2*c1_f2+c0_f2*c2_f05)

        With a*a=1 and b*b=4 we would make c0_f05 more important:

        score = 6.0 *(c0_f2 * c1_f2 * c2_f05)/(c1_f2*c2_f05+c0_f2*c2_f05+4.0*c0_f2*c1_f2)
        """
        # Set user attributes for Optuna tracking
        macro_f1 = float(lovo_results["aggregation"]["macro_group_mean"]["macro_f1"])
        trial.set_user_attr("macro_f1", macro_f1)
        trial.set_user_attr("class_0_f1", c0_f1)
        trial.set_user_attr("class_1_f1", c1_f1)
        trial.set_user_attr("class_2_f1", c2_f1)
        trial.set_user_attr("class2_threshold", c2_threshold)
        trial.set_user_attr("custom_rate_score", score)

        return score

    # Run Optuna optimization loop

    # Calculate remaining trials so n_trials acts as a lifetime target
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining_trials = max(0, n_trials - completed_trials)

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, gc_after_trial=True)
    else:
        logger.info("Study already has %d completed trials (target: %d). Skipping optimization.", completed_trials, n_trials)

    # Compile results payload
    best_attrs = study.best_trial.user_attrs
    output_payload = {
        "study_name": study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_metrics": {
            "macro_f1": best_attrs.get("macro_f1", 0.0),
            "class_0_f1": best_attrs.get("class_0_f1", 0.0),
            "class_1_f1": best_attrs.get("class_1_f1", 0.0),
            "class_2_f1": best_attrs.get("class_2_f1", 0.0),
        },
    }

    # Mirror output to screen
    logger.info("=" * 60)
    logger.info("HPO OPTIMIZATION COMPLETE")
    logger.info("Best Objective Score: %.4f", study.best_value)
    logger.info("Best Hyperparameters:")
    for k, v in study.best_params.items():
        logger.info("  - %s: %s", k, v)
    logger.info("Best Metrics -> Macro F1: %.4f | C0 F1: %.4f | C1 F1: %.4f | C2 F1: %.4f",
                output_payload["best_metrics"]["macro_f1"],
                output_payload["best_metrics"]["class_0_f1"],
                output_payload["best_metrics"]["class_1_f1"],
                output_payload["best_metrics"]["class_2_f1"])
    logger.info("=" * 60)

    # Save output mirror json to output/hpo/
    output_dir = Path("output/hpo")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{study_name}_best_params.json"
    
    with open(json_path, "w") as f:
        json.dump(output_payload, f, indent=4)
        
    logger.info("Successfully exported winning configuration JSON to %s", json_path)

    # Execute Final Production Master Refit using winning parameters
    log_msg = (
            f"Starting Final Production Master Refit with best parameters..."
    )

    # Let the logger do its job if INFO is enabled
    logger.info(log_msg)

    # If the logger is silenced at WARNING or above, print directly to the console
    if logger.getEffectiveLevel() >= logging.WARNING:
            print(log_msg)

    resolved_params = dict(study.best_params)
    for name, spec in search_space.items():
        if "fixed" in spec:
            resolved_params[name] = spec["fixed"]

    final_settings = Config(settings).with_overrides({
        "training.lr": resolved_params["lr"],
        "training.weight_decay": resolved_params["weight_decay"],
        "training.dropout": resolved_params["dropout"],
        "training.class_weights": (
            resolved_params["class_weight_0"],
            1.0,
            resolved_params["class_weight_2"],
        ),
        "training.c2_threshold": resolved_params["c2_threshold"],
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
 
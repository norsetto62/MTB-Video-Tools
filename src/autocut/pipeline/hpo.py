import json
import logging
from pathlib import Path
import copy
from dataclasses import replace
import numpy as np
import optuna
import yaml
from datetime import datetime

from ..config_loader import TrainingRunConfig
from .lovo import run_lovo  
from ..dataset.manifest import load_manifest
from .lovo import run_lovo
from .training import _combine_records
from ..ml.training import train_fixed_epochs

logger = logging.getLogger(__name__)


def run_hpo(
    manifest_path: str | Path,
    settings: TrainingRunConfig,
) -> None:
    """
    Orchestrates the Optuna HPO study, logs outputs to screen, 
    and exports winning parameters to output/hpo/.
    """
    
    # Database path explicitly placed in data/db/
    db_dir = Path("data/db")
    db_dir.mkdir(parents=True, exist_ok=True)
    storage_db = f"sqlite:///{db_dir / 'hpo_study.db'}"
    
    n_trials = settings.n_trials
    proxy_groups = settings.proxy_test_groups
    study_name = settings.study_name

    logger.info("Initializing Optuna study '%s' with storage: %s", study_name, storage_db)
    logger.info("Proxy validation folds assigned for tuning: %s", proxy_groups)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_db,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings.training.seed),
    )

    search_space = settings.hpo_search_space
    if not search_space:
        raise ValueError(
            "HPO search space is empty. Configure hpo.search_space in the training YAML."
        )

    def suggest_parameters(trial: optuna.Trial) -> dict[str, object]:
        """Build Optuna suggestions from the validated YAML distributions."""
        params: dict[str, object] = {}
        for name, spec in search_space.items():
            if "choices" in spec:
                params[name] = trial.suggest_categorical(name, spec["choices"])
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

        filtered_train_groups = [g for g in settings.train_groups if g not in proxy_groups]

        # Construct an updated training sub-config and main config safely using replace()
        updated_training = replace(
            settings.training,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            class_weights=class_weights,
            hidden_dim=hidden_dim,
        )
        
        trial_settings = replace(
            settings,
            training=updated_training,
            train_groups=filtered_train_groups,
            validation_groups=proxy_groups, # Pass proxy groups as tuple/list expected by our config schema
        )

        # Execute proxy evaluation across subset folds
        lovo_results = run_lovo(
            manifest_path=manifest_path,
            settings=trial_settings
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
        total_true_0 = max(1.0, np.sum(total_cm[0]))
        total_true_1 = max(1.0, np.sum(total_cm[1]))
        total_true_2 = max(1.0, np.sum(total_cm[2]))

        # 1. TARGET METRICS (Class 2 is what we want)
        class_2_recall = total_cm[2][2] / total_true_2     # Catch as many real Class 2s as possible

        # 2. FALSE TRIGGER RATES FROM BACKGROUND (Class 0 & 1)
        rate_0_to_2 = total_cm[0][2] / total_true_0  # DISASTER: Class 0 guessed as Class 2
        rate_1_to_2 = total_cm[1][2] / total_true_1  # BAD: Class 1 guessed as Class 2

        # 3. Multiplicative Scaling
        score = float(class_2_recall * (1.0 - rate_0_to_2) * (1.0 - rate_1_to_2))

        # Set user attributes for Optuna tracking
        macro_f1 = float(lovo_results["aggregation"]["macro_group_mean"]["macro_f1"])
        trial.set_user_attr("macro_f1", macro_f1)
        trial.set_user_attr("class_0_f1", c0_f1)
        trial.set_user_attr("class_1_f1", c1_f1)
        trial.set_user_attr("class_2_f1", c2_f1)
        trial.set_user_attr("class_2_recall", class_2_recall)
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
    
    best_params = study.best_params
    
    final_training = replace(
        settings.training,
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        dropout=best_params["dropout"],
        class_weights=[best_params["class_weight_0"], 1.0, best_params["class_weight_2"]],
        hidden_dim=best_params["hidden_dim"],
    )

    final_settings = replace(
        settings,
        training=final_training,
    )

    result = run_lovo(manifest_path, final_settings)
    # Determine the optimal epoch count for the final master refit 
    # (e.g., taking the mean or median of the consensus selected epochs from each fold)
    fold_epochs = [
        fold.get("consensus_selected_epochs") or fold.get("selected_epochs") 
        for fold in result["folds"]
    ]
    # Fallback to settings.training.epochs if epochs aren't explicitly tracked per fold
    mean_epochs = int(np.mean([e for e in fold_epochs if e is not None])) if fold_epochs else settings.training.epochs

    print(f"Refitting master production model on all groups for {mean_epochs} epochs...")

    # Load and combine all records using the codebase's native helper
    records = load_manifest(manifest_path, verify_files=True)
    all_windows, all_targets = _combine_records(records, settings)

    # Define your production checkpoint destination
    base_exp_name = getattr(settings, "experiment_name", "baseline")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_exp_name = f"{base_exp_name}_{timestamp}"
    models_dir = Path(settings.common.models_dir) / "experiments"
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
 
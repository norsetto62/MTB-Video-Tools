import argparse
from pathlib import Path
from math import dist

import optuna

def main():
    parser = argparse.ArgumentParser(
        description="Plot Optuna study results."
    )
    parser.add_argument(
        "study_name",
        help="Name of the Optuna study to plot, e.g. study_1",
    )
    parser.add_argument(
        "--database",
        default="data/db/hpo_study.db",
        help="Path to the Optuna SQLite database",
    )
    args = parser.parse_args()

    db_path = Path(args.database).resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Optuna database not found: {db_path}")

    storage_url = f"sqlite:///{db_path.as_posix()}"

    study = optuna.load_study(
        study_name=args.study_name,
        storage=storage_url,
    )

    # 1. First, check if any completed trials exist at all
    print(f"Study: {study.study_name}")
    print(f"Trials: {len(study.trials)}")
    if len(study.trials) == 0:
        print("No completed trials yet.")
    else:
        # 2. Check if the study handles multiple objectives
        if len(study.directions) > 1:
            print(f"Multi-objective study detected. Found {len(study.best_trials)} Pareto-optimal trials:\n")

            cid = min(
                    study.best_trials, 
                    key=lambda t: dist([t.values[0], t.values[1]], [1.0, 0.0])
                    )
            lg = max(
                    study.best_trials, 
                    key=lambda t: t.values[0] - t.values[1]
                    )
            hpr = min(
                    study.best_trials,
                    key=lambda t: (-t.values[0], t.values[1], t.number,)
                    )
            print(
                f" Closest ideal point     : {cid.number} with parameters: {cid.params}\n", 
                f"Largest gap             : {lg.number} with parameters: {lg.params}\n",
                f"Highest Positive Recall : {hpr.number}  with parameters: {hpr.params}\n",
            )        
            for i, trial in enumerate(study.best_trials, 1):
                print(f"  Pareto Trial #{i} (Trial {trial.number}):")
                print(f"    Values: {[round(v, 6) for v in trial.values]}")
                print(f"    Params: {trial.params}")
            try:
                optuna.visualization.plot_pareto_front(study).show()
            except (ValueError, RuntimeError) as exc:
                print(f"Could not plot Pareto front: {exc}")
            
        else:
            # 3. Fallback to single-objective logic safely
            if study.best_trial is not None:
                print(f"Best value: {study.best_value:.6f}")
                print(f"Best parameters: {study.best_params}")
                try:
                    optuna.visualization.plot_optimization_history(study).show()
                except (ValueError, RuntimeError) as exc:
                    print(f"Could not plot optimization history: {exc}")
                try:
                        optuna.visualization.plot_slice(study).show()
                except (ValueError, RuntimeError) as exc:
                    print(f"Could not plot slice: {exc}")
                try:
                    optuna.visualization.plot_param_importances(study).show()
                except (ValueError, RuntimeError) as exc:
                    print(f"Could not plot parameter importances: {exc}")

if __name__ == "__main__":
    main()


import argparse
from pathlib import Path

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

    print(f"Study: {study.study_name}")
    print(f"Trials: {len(study.trials)}")
    if study.best_trial is not None:
        print(f"Best value: {study.best_value:.6f}")
        print(f"Best parameters: {study.best_params}")
    else:
        print("No completed trials yet.")

    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_slice(study).show()

    try:
        optuna.visualization.plot_param_importances(study).show()
    except (ValueError, RuntimeError) as exc:
        print(f"Could not plot parameter importances: {exc}")


if __name__ == "__main__":
    main()

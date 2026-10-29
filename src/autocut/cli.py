"""Command-line parsing and dispatch for AutoCut."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Sequence
from datetime import datetime

COMMANDS = ("run", "audit", "train", "validate", "test", "lovo", "lightgbm-train", "lightgbm-lovo", "lightgbm-hpo", "hpo", "models")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autocut",
        description="Analyse MTB footage and assemble highlight edits.",
        epilog=(
            "For compatibility, 'autocut annotations.txt' is equivalent to "
            "'autocut run annotations.txt'."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Generate a highlight edit.")
    run.add_argument("annotations", type=Path, help="Annotation TXT file.")
    run.add_argument("--config", type=Path, default=Path("config/autocut.yaml"))
    run.add_argument("--output", type=Path, help="Output video path (default: output/<annotations>_autocut.mp4).")
    run.add_argument(
        "--video-encoder",
        choices=("h264_nvenc", "libx264"),
        help="Override video.encoder from the run configuration.",
    )

    audit = subparsers.add_parser("audit", help="Audit annotations and derived data.")
    audit.add_argument("source", type=Path, help="Annotation TXT or dataset manifest.")
    audit.add_argument("--config", type=Path, default=Path("config/autocut.yaml"))
    audit.add_argument("--training-config", type=Path, default=Path("config/train.yaml"))

    train = subparsers.add_parser("train", help="Train a model from a fixed group split.")
    train.add_argument("manifest", type=Path, nargs="?", help="Dataset manifest (overrides YAML).")
    train.add_argument("--config", type=Path, default=Path("config/train.yaml"))

    validate = subparsers.add_parser("validate", help="Evaluate a configured holdout split.")
    validate.add_argument("manifest", type=Path, nargs="?", help="Dataset manifest (overrides YAML).")
    validate.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    validate.add_argument("--checkpoint", type=Path, required=True, help="Saved model checkpoint.")
    validate.add_argument("--group", required=True, help="Manifest group to evaluate.")
    validate.add_argument("--output", type=Path, help="Prediction JSON path (default: results directory).")
    validate.add_argument("--batch-size", type=int, default=256, help="Inference batch size.")

    test = subparsers.add_parser(
        "test",
        help="Run pytest and return its exit status.",
        add_help=False,
    )
    # Pytest owns its flags (including -h/--help), so don't let argparse's
    # subparser option parsing consume or reject them.
    test.add_argument("pytest_args", nargs=argparse.REMAINDER)

    lovo = subparsers.add_parser("lovo", help="Run legacy P/U leave-one-group-out evaluations.")
    lovo.add_argument("manifest", type=Path, nargs="?", help="Dataset manifest (overrides YAML).")
    lovo.add_argument("--config", type=Path, default=Path("config/train.yaml"))

    lightgbm_train = subparsers.add_parser(
        "lightgbm-train",
        help="Train one LightGBM ranking model on annotated windows.",
    )
    lightgbm_train.add_argument("manifest", type=Path, nargs="?", help="Dataset manifest (overrides YAML).")
    lightgbm_train.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    lightgbm_train.add_argument("--checkpoint", type=Path, help="Override the single-model checkpoint path.")

    lightgbm_report = subparsers.add_parser(
        "lightgbm-score-report",
        help="Score every window and export per-video/global LightGBM ranking diagnostics.",
    )
    lightgbm_report.add_argument("manifest", type=Path, nargs="?", help="Dataset manifest (overrides YAML).")
    lightgbm_report.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    lightgbm_report.add_argument("--checkpoint", type=Path, help="Override the LightGBM ranker checkpoint.")
    lightgbm_report.add_argument(
        "--output", type=Path, default=Path("output/lightgbm_window_scores.csv"),
        help="Primary global-score CSV path (companion video-range CSV is written beside it).",
    )

    lightgbm_lovo = subparsers.add_parser(
        "lightgbm-lovo",
        help="Train and validate a LightGBM ranking model by manifest group.",
    )
    lightgbm_lovo.add_argument("manifest", type=Path, nargs="?", help="Dataset manifest (overrides YAML).")
    lightgbm_lovo.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    lightgbm_lovo.add_argument("--checkpoint-dir", type=Path, help="Override the LightGBM LOVO checkpoint directory.")

    lightgbm_hpo = subparsers.add_parser(
        "lightgbm-hpo",
        help="Tune LightGBM ranking objective and hyperparameters with Optuna.",
    )
    lightgbm_hpo.add_argument(
        "--config", type=Path, default=Path("config/train.yaml"),
        help="Training and LightGBM HPO YAML configuration file",
    )
    lightgbm_hpo.add_argument("manifest", type=Path, nargs="?", help="Dataset manifest (overrides YAML).")

    models = subparsers.add_parser("models", help="Inspect a saved checkpoint.")
    models.add_argument("checkpoint", type=Path, nargs="?", help="Checkpoint to inspect.")
    models.add_argument("--config", type=Path, default=Path("config/autocut.yaml"))

    hpo = subparsers.add_parser(
        "hpo",
        help="Run Optuna hyperparameter optimization",
    )
    hpo.add_argument(
        "--config", 
        type=Path, 
        default=Path("config/train.yaml"), 
        help="Training and HPO YAML configuration file"
    )
    hpo.add_argument("manifest", type=Path, nargs="?", help="Dataset manifest (overrides YAML).")
    hpo.add_argument(
        "--diagnose-trial",
        type=int,
        help="Export full-timeline selection diagnostics for an existing trial without running new trials or a production refit.",
    )
    hpo.add_argument(
        "--diagnostics-output",
        type=Path,
        help="CSV output path for --diagnose-trial (default: output/hpo/<study>_trial_<N>_selection_diagnostics.csv).",
    )
    
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _unimplemented(command: str) -> int:
    print(
        f"AutoCut '{command}' orchestration is not implemented yet. "
        "The CLI and configuration validation are available, but this command "
        "cannot run end-to-end until its Phase 12 pipeline handler is added.",
        file=sys.stderr,
    )
    return 2


def _run_tests(pytest_args: Sequence[str]) -> int:
    command = [sys.executable, "-m", "pytest", *pytest_args]
    try:
        completed = subprocess.run(command, check=False)
    except OSError as exc:
        print(f"Unable to start pytest: {exc}", file=sys.stderr)
        return 2
    return completed.returncode


def _inspect_checkpoint(checkpoint: Path) -> int:
    from .ml.checkpoint import load_checkpoint, validate_checkpoint

    if not checkpoint.is_file():
        print(f"Checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2
    try:
        payload = validate_checkpoint(load_checkpoint(checkpoint, map_location="cpu"))
    except Exception as exc:
        print(f"Invalid or unreadable checkpoint {checkpoint}: {exc}", file=sys.stderr)
        return 2

    summary = {
        "path": str(checkpoint),
        "checkpoint_version": payload.get("checkpoint_version"),
        "feature_names": payload.get("feature_names"),
        "feature_mask": payload.get("feature_mask"),
        "class_mapping": payload.get("class_mapping"),
        "training_config": payload.get("training_config"),
        "data_config": payload.get("data_config"),
        "epoch": payload.get("epoch"),
        "best_metric": payload.get("best_metric"),
        "metadata": payload.get("metadata"),
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


def dispatch(args: argparse.Namespace) -> int:
    """Dispatch a parsed command, validating its configuration first."""
    if args.command == "test":
        return _run_tests(args.pytest_args)

    if args.command == "models":
        checkpoint = args.checkpoint
        if checkpoint is None:
            from collections.abc import Mapping

            from .config_loader import load_config

            config = load_config(args.config)
            model = config.get("model", {})
            if not isinstance(model, Mapping):
                raise ValueError("Configuration section 'model' must be a mapping.")
            configured_checkpoint = model.get("checkpoint", "models/best_model.pt")
            if (
                not isinstance(configured_checkpoint, (str, Path))
                or not str(configured_checkpoint).strip()
            ):
                raise ValueError("model.checkpoint must be a non-empty path string.")
            checkpoint = Path(configured_checkpoint).expanduser()
        return _inspect_checkpoint(checkpoint)

    if args.command == "audit":
        from .config_loader import load_config
        from .dataset.audit import audit_source, print_audit_report

        if not args.source.is_file():
            print(f"Input file not found: {args.source}", file=sys.stderr)
            return 2
        settings = load_config(args.config)
        logging_section = settings.get("logging", {})
        if not hasattr(logging_section, "get"):
            raise ValueError("Configuration section 'logging' must be a mapping.")
        _configure_logging(logging_section.get("level", "INFO"))
        report = audit_source(
            args.source,
            settings,
            training_config=args.training_config,
        )
        print_audit_report(report)
        return 1 if report["status"] == "error" else 0

    if args.command == "run":
        from .config_loader import load_config
        from .pipeline.run import run_pipeline

        settings = load_config(args.config)
        logging_section = settings.get("logging", {})
        if not isinstance(logging_section, dict) and not hasattr(logging_section, "get"):
            raise ValueError("Configuration section 'logging' must be a mapping.")
        _configure_logging(logging_section.get("level", "INFO"))
        if not args.annotations.is_file():
            print(f"Input file not found: {args.annotations}", file=sys.stderr)
            return 2
        result = run_pipeline(
            args.annotations,
            settings,
            output_path=args.output,
            video_encoder=args.video_encoder,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.command == "train":
        from .config_loader import load_config
        from .pipeline.training import run_training

        settings = load_config(args.config)
        logging_section = settings.get("logging", {})
        if not hasattr(logging_section, "get"):
            raise ValueError("Configuration section 'logging' must be a mapping.")
        _configure_logging(logging_section.get("level", "INFO"))
        data = settings.get("data", {})
        if not hasattr(data, "get"):
            raise ValueError("Configuration section 'data' must be a mapping.")
        manifest = args.manifest or Path(
            data.get("manifest", "data/manifest/annotations_manifest.yaml")
        ).expanduser()
        if not manifest.is_file():
            print(f"Dataset manifest not found: {manifest}", file=sys.stderr)
            return 2
        result = run_training(manifest, settings)
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.command == "lightgbm-train":
        from .config_loader import load_config
        from .pipeline.lightgbm_training import run_lightgbm_train_from_manifest

        settings = load_config(args.config)
        logging_section = settings.get("logging", {})
        if not hasattr(logging_section, "get"):
            raise ValueError("Configuration section 'logging' must be a mapping.")
        _configure_logging(logging_section.get("level", "INFO"))
        data = settings.get("data", {})
        if not hasattr(data, "get"):
            raise ValueError("Configuration section 'data' must be a mapping.")
        manifest = args.manifest or Path(
            data.get("manifest", "data/manifest/annotations_manifest.yaml")
        ).expanduser()
        if not manifest.is_file():
            print(f"Dataset manifest not found: {manifest}", file=sys.stderr)
            return 2
        result = run_lightgbm_train_from_manifest(
            manifest,
            settings,
            checkpoint_path=args.checkpoint,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.command == "lightgbm-score-report":
        from .config_loader import load_config
        from .pipeline.lightgbm_score_report import run_lightgbm_score_report

        settings = load_config(args.config)
        logging_section = settings.get("logging", {})
        if not hasattr(logging_section, "get"):
            raise ValueError("Configuration section 'logging' must be a mapping.")
        _configure_logging(logging_section.get("level", "INFO"))
        data = settings.get("data", {})
        if not hasattr(data, "get"):
            raise ValueError("Configuration section 'data' must be a mapping.")
        manifest = args.manifest or Path(
            data.get("manifest", "data/manifest/annotations_manifest.yaml")
        ).expanduser()
        if not manifest.is_file():
            print(f"Dataset manifest not found: {manifest}", file=sys.stderr)
            return 2
        result = run_lightgbm_score_report(
            manifest, settings, checkpoint_path=args.checkpoint, output_path=args.output
        )
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.command == "lightgbm-lovo":
        from .config_loader import load_config
        from .pipeline.lightgbm_training import run_lightgbm_lovo_from_manifest

        settings = load_config(args.config)
        logging_section = settings.get("logging", {})
        if not hasattr(logging_section, "get"):
            raise ValueError("Configuration section 'logging' must be a mapping.")
        _configure_logging(logging_section.get("level", "INFO"))
        data = settings.get("data", {})
        if not hasattr(data, "get"):
            raise ValueError("Configuration section 'data' must be a mapping.")
        manifest = args.manifest or Path(
            data.get("manifest", "data/manifest/annotations_manifest.yaml")
        ).expanduser()
        if not manifest.is_file():
            print(f"Dataset manifest not found: {manifest}", file=sys.stderr)
            return 2
        results = run_lightgbm_lovo_from_manifest(
            manifest,
            settings,
            checkpoint_dir=args.checkpoint_dir,
        )
        print(json.dumps([
            {
                "held_out_group": result.held_out_group,
                "train_video_ids": result.train_video_ids,
                "validation_video_ids": result.validation_video_ids,
                **result.summary,
            }
            for result in results
        ], indent=2, default=str))
        return 0

    if args.command in {"validate", "lovo", "hpo", "lightgbm-hpo"}:
        from .config_loader import load_config

        settings = load_config(args.config)
        logging_section = settings.get("logging", {})
        if not hasattr(logging_section, "get"):
            raise ValueError("Configuration section 'logging' must be a mapping.")
        _configure_logging(logging_section.get("level", "INFO"))
        data = settings.get("data", {})
        if not hasattr(data, "get"):
            raise ValueError("Configuration section 'data' must be a mapping.")
        manifest = args.manifest or Path(
            data.get("manifest", "data/manifest/annotations_manifest.yaml")
        ).expanduser()
        if not manifest.is_file():
            print(f"Dataset manifest not found: {manifest}", file=sys.stderr)
            return 2
        if args.command == "lightgbm-hpo":
            from .pipeline.lightgbm_hpo import run_lightgbm_hpo

            result = run_lightgbm_hpo(manifest, settings)
            print(json.dumps(result, indent=2, default=str))
            return 0
        if args.command == "hpo":
            if args.diagnose_trial is not None:
                from .pipeline.hpo import run_hpo_diagnostics

                report_path = run_hpo_diagnostics(
                    manifest,
                    settings,
                    trial_number=args.diagnose_trial,
                    output_path=args.diagnostics_output,
                )
                print(f"Selection diagnostics written to: {report_path.resolve()}")
                return 0
            if args.diagnostics_output is not None:
                raise ValueError("--diagnostics-output requires --diagnose-trial.")
            from .pipeline.hpo import run_hpo

            run_hpo(manifest, settings)
            return 0
        if args.command == "validate":
            from .pipeline.validation import run_validation

            result = run_validation(
                manifest,
                settings,
                checkpoint_path=args.checkpoint,
                group_id=args.group,
                output_path=args.output,
                batch_size=args.batch_size,
            )
            print(json.dumps(result, indent=2, default=str))
            return 0
        # LOVO
        from .dataset.manifest import load_manifest
        from .pipeline.lovo import run_lovo
        from .pipeline.training import _combine_records
        from .ml.training import train_fixed_epochs
        from .pipeline.training import _training_config
        import numpy as np

        result = run_lovo(manifest, settings)
        print(json.dumps(result, indent=2, default=str))
        # Determine the optimal epoch count for the final master refit 
        # (e.g., taking the mean or median of the consensus selected epochs from each fold)
        fold_epochs = [
            fold.get("consensus_selected_epochs") or fold.get("selected_epochs") 
            for fold in result["folds"]
        ]
        # Fallback to configured training epochs if folds omit them
        mean_epochs = int(np.mean([e for e in fold_epochs if e is not None])) if fold_epochs else _training_config(settings).epochs

        print(f"Refitting master production model on all groups for {mean_epochs} epochs...")

        # Load and combine all records using the codebase's native helper
        records = load_manifest(manifest, verify_files=True)
        all_windows, all_targets = _combine_records(records, settings)

        # Define your production checkpoint destination
        experiment = settings.get("experiment", {})
        base_exp_name = experiment.get("name", "baseline")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_exp_name = f"{base_exp_name}_{timestamp}"
        paths = settings.get("paths", {})
        models_dir = Path(paths.get("models_dir", "models")).expanduser() / "experiments"
        models_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = models_dir / f"{unique_exp_name}_master_production.pt"

        # Train and save using your exact fixed-epoch function
        model, scaler, train_summary = train_fixed_epochs(
            train_windows=all_windows,
            train_targets=all_targets,
            config=_training_config(settings),
            epochs=mean_epochs,
            checkpoint_path=checkpoint_path,
            data_config=result.get("data_config"),
            metadata={"experiment_name": base_exp_name},
        )

        print(f"Master production model successfully saved to {checkpoint_path}")
        return 0

    print(f"Unsupported command: {args.command}", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; bare annotation paths retain the legacy shorthand."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] not in COMMANDS and arguments[0] not in {"-h", "--help"}:
        arguments.insert(0, "run")
    # Pass pytest's arguments through untouched: argparse subparsers otherwise
    # reject flags such as -q before a REMAINDER positional can capture them.
    if arguments and arguments[0] == "test":
        return _run_tests(arguments[1:])

    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        return dispatch(args)
    except (ValueError, FileNotFoundError, TypeError, RuntimeError, OSError) as exc:
        print(f"Configuration/input error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

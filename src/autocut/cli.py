"""Command-line parsing and dispatch for AutoCut."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Sequence

COMMANDS = ("run", "audit", "train", "test", "score-report", "lovo", "hpo", "models")


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
        "--checkpoint", type=Path,
        help="Override model checkpoint (for LightGBM use models/lightgbm/ranker.json).",
    )
    run.add_argument(
        "--video-encoder",
        choices=("h264_nvenc", "libx264"),
        help="Override video.encoder from the run configuration.",
    )

    audit = subparsers.add_parser("audit", help="Audit annotations and derived data.")
    audit.add_argument("source", type=Path, help="Annotation TXT or dataset manifest.")
    audit.add_argument("--config", type=Path, default=Path("config/autocut.yaml"))
    audit.add_argument("--training-config", type=Path, default=Path("config/train.yaml"))

    train = subparsers.add_parser("train", help="Train the LightGBM ranking model from a fixed group split.")
    train.add_argument("manifest", type=Path, nargs="?", help="Dataset manifest (overrides YAML).")
    train.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    train.add_argument("--checkpoint", type=Path, help="Override the LightGBM ranker checkpoint.")

    test = subparsers.add_parser(
        "test",
        help="Run pytest and return its exit status.",
        add_help=False,
    )
    # Pytest owns its flags (including -h/--help), so don't let argparse's
    # subparser option parsing consume or reject them.
    test.add_argument("pytest_args", nargs=argparse.REMAINDER)

    score_report = subparsers.add_parser(
        "score-report",
        help="Score every window and export per-video/global LightGBM ranking diagnostics.",
    )
    score_report.add_argument("manifest", type=Path, nargs="?", help="Dataset manifest (overrides YAML).")
    score_report.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    score_report.add_argument("--checkpoint", type=Path, help="Override the LightGBM ranker checkpoint.")
    score_report.add_argument(
        "--output", type=Path, default=Path("output/lightgbm_window_scores.csv"),
        help="Primary global-score CSV path (companion video-range CSV is written beside it).",
    )

    lovo = subparsers.add_parser(
        "lovo",
        help="Train and validate a LightGBM ranking model by manifest group.",
    )
    lovo.add_argument("manifest", type=Path, nargs="?", help="Dataset manifest (overrides YAML).")
    lovo.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    lovo.add_argument("--checkpoint-dir", type=Path, help="Override the LightGBM LOVO checkpoint directory.")

    hpo = subparsers.add_parser(
        "hpo",
        help="Tune LightGBM ranking objective and hyperparameters with Optuna.",
    )
    hpo.add_argument(
        "--config", type=Path, default=Path("config/train.yaml"),
        help="Training and LightGBM HPO YAML configuration file",
    )
    hpo.add_argument("manifest", type=Path, nargs="?", help="Dataset manifest (overrides YAML).")

    models = subparsers.add_parser("models", help="Inspect a saved checkpoint.")
    models.add_argument("checkpoint", type=Path, nargs="?", help="Checkpoint to inspect.")
    models.add_argument("--config", type=Path, default=Path("config/autocut.yaml"))

    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _run_tests(pytest_args: Sequence[str]) -> int:
    command = [sys.executable, "-m", "pytest", *pytest_args]
    try:
        completed = subprocess.run(command, check=False)
    except OSError as exc:
        print(f"Unable to start pytest: {exc}", file=sys.stderr)
        return 2
    return completed.returncode


def _inspect_checkpoint(checkpoint: Path) -> int:
    from .ml.lightgbm_checkpoint import load_lightgbm_checkpoint

    if not checkpoint.is_file():
        print(f"Checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2
    try:
        payload = load_lightgbm_checkpoint(checkpoint)
    except Exception as exc:
        print(f"Invalid or unreadable LightGBM checkpoint {checkpoint}: {exc}", file=sys.stderr)
        return 2
    summary = {
        "path": str(checkpoint),
        "model_class": payload.get("model_class"),
        "feature_names": payload.get("feature_names"),
        "data_config": payload.get("data_config"),
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
            configured_checkpoint = model.get("checkpoint", "models/lightgbm/ranker.json")
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
            args.annotations, settings, output_path=args.output,
            video_encoder=args.video_encoder, checkpoint_path=args.checkpoint,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.command == "train":
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

    if args.command == "score-report":
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

    if args.command == "lovo":
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

    if args.command == "hpo":
        from .config_loader import load_config
        from .pipeline.lightgbm_hpo import run_lightgbm_hpo

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
        result = run_lightgbm_hpo(manifest, settings)
        print(json.dumps(result, indent=2, default=str))
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

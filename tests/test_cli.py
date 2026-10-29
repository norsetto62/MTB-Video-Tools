from pathlib import Path

from autocut.cli import build_parser, main


def test_help_lists_commands(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out
    for command in ("run", "audit", "train", "validate", "test", "lovo", "lightgbm-train", "lightgbm-lovo", "lightgbm-hpo", "models"):
        assert command in output


def test_bare_annotation_path_defaults_to_run():
    args = build_parser().parse_args(["run", "annotations.txt"])
    assert args.command == "run"
    assert str(args.annotations) == "annotations.txt"


def test_test_command_forwards_pytest_arguments(monkeypatch):
    from autocut import cli
    captured = {}

    class Result:
        returncode = 7

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check
        return Result()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert main(["test", "-q", "tests/test_config.py"]) == 7
    assert captured["command"][-2:] == ["-q", "tests/test_config.py"]
    assert captured["check"] is False


def test_cli_lovo_dispatch_calls_pipeline(tmp_path, monkeypatch, capsys):
    from autocut import cli
    from autocut.config_loader import Config

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("videos: []\n", encoding="utf-8")
    settings = Config({
        "paths": {"models_dir": str(tmp_path / "models")},
        "data": {
            "manifest": str(manifest),
            "train_groups": ["train"],
            "validation_groups": ["val"],
        },
        "training": {"epochs": 2},
        "experiment": {"name": "cli-test"},
        "logging": {"level": "INFO"},
    })
    config_path = tmp_path / "train.yaml"
    config_path.write_text("logging:\n  level: INFO\n", encoding="utf-8")
    monkeypatch.setattr("autocut.config_loader.load_config", lambda path: settings)
    captured = {}

    def fake_run_lovo(path, loaded_settings):
        captured["manifest"] = path
        captured["settings"] = loaded_settings
        return {
            "fold_count": 3,
            "folds": [
                {"selected_epochs": 2},
                {"selected_epochs": 4},
                {"selected_epochs": 6},
            ],
            "data_config": {"flow_fps": 2.0},
        }

    monkeypatch.setattr("autocut.pipeline.lovo.run_lovo", fake_run_lovo)
    monkeypatch.setattr(
        "autocut.dataset.manifest.load_manifest",
        lambda path, verify_files=True: ["record"],
    )
    monkeypatch.setattr(
        "autocut.pipeline.training._combine_records",
        lambda records, loaded_settings: ([], []),
    )
    monkeypatch.setattr(
        "autocut.ml.training.train_fixed_epochs",
        lambda **kwargs: (None, None, {"epochs_trained": kwargs["epochs"]}),
    )
    assert main(["lovo", str(manifest), "--config", str(config_path)]) == 0
    assert captured["manifest"] == manifest
    assert captured["settings"] is settings
    assert '"fold_count": 3' in capsys.readouterr().out


def test_cli_run_dispatch_loads_generic_config_and_calls_pipeline(tmp_path, monkeypatch, capsys):
    from autocut.config_loader import Config

    annotations = tmp_path / "annotations.txt"
    annotations.write_text("placeholder", encoding="utf-8")
    config = tmp_path / "autocut.yaml"
    config.write_text("logging:\n  level: INFO\n", encoding="utf-8")
    captured = {}

    def fake_run_pipeline(path, loaded_settings, *, output_path=None, video_encoder=None):
        captured["path"] = path
        captured["settings"] = loaded_settings
        captured["output_path"] = output_path
        captured["video_encoder"] = video_encoder
        return {"status": "completed"}

    monkeypatch.setattr("autocut.pipeline.run.run_pipeline", fake_run_pipeline)
    output = tmp_path / "custom.mp4"
    assert main([
        "run", str(annotations), "--config", str(config),
        "--output", str(output),
    ]) == 0
    assert captured["path"] == annotations
    assert isinstance(captured["settings"], Config)
    assert captured["settings"].logging.level == "INFO"
    assert captured["output_path"] == output
    assert captured["video_encoder"] is None
    assert '"status": "completed"' in capsys.readouterr().out



def test_hpo_cli_defaults_to_training_config():
    args = build_parser().parse_args(["hpo"])
    assert args.command == "hpo"
    assert args.config == Path("config/train.yaml")


def test_models_command_loads_checkpoint_from_generic_config(tmp_path, monkeypatch):
    config = tmp_path / "minimal.yaml"
    checkpoint = tmp_path / "trained" / "model.pt"
    config.write_text(
        f"model:\n  checkpoint: {checkpoint.as_posix()}\n",
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(
        "autocut.cli._inspect_checkpoint",
        lambda path: captured.update(checkpoint=path) or 0,
    )

    assert main(["models", "--config", str(config)]) == 0
    assert captured["checkpoint"] == checkpoint


def test_models_command_rejects_non_mapping_model_section(tmp_path, capsys):
    config = tmp_path / "bad.yaml"
    config.write_text("model: invalid\n", encoding="utf-8")

    assert main(["models", "--config", str(config)]) == 2
    assert "Configuration section 'model' must be a mapping" in capsys.readouterr().err


def test_models_command_explicit_checkpoint_skips_config_loading(tmp_path, monkeypatch):
    from autocut import cli

    checkpoint = tmp_path / "explicit.pt"
    monkeypatch.setattr(
        cli,
        "_inspect_checkpoint",
        lambda path: 0 if path == checkpoint else 99,
    )
    monkeypatch.setattr(
        "autocut.config_loader.load_config",
        lambda path: (_ for _ in ()).throw(AssertionError("config should not be loaded")),
    )

    assert main(["models", str(checkpoint), "--config", str(tmp_path / "unused.yaml")]) == 0



def test_lightgbm_train_cli_dispatches_to_single_model_pipeline(tmp_path, monkeypatch, capsys):
    from autocut.config_loader import Config

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("videos: []\n", encoding="utf-8")
    config_path = tmp_path / "train.yaml"
    config_path.write_text("logging:\n  level: INFO\n", encoding="utf-8")
    settings = Config({
        "data": {"manifest": str(manifest)},
        "logging": {"level": "INFO"},
    })
    monkeypatch.setattr("autocut.config_loader.load_config", lambda path: settings)
    captured = {}
    checkpoint = tmp_path / "models" / "single.json"

    def fake_train(path, loaded_settings, *, checkpoint_path=None):
        captured.update(path=path, settings=loaded_settings, checkpoint_path=checkpoint_path)
        return {"checkpoint_path": str(checkpoint), "training_rows": 12}

    monkeypatch.setattr(
        "autocut.pipeline.lightgbm_training.run_lightgbm_train_from_manifest",
        fake_train,
    )
    assert main([
        "lightgbm-train", "--config", str(config_path), "--checkpoint", str(checkpoint)
    ]) == 0
    assert captured == {
        "path": manifest,
        "settings": settings,
        "checkpoint_path": checkpoint,
    }
    assert '"training_rows": 12' in capsys.readouterr().out

from pathlib import Path

from autocut.cli import build_parser, main
from autocut.config_loader import Config
from autocut.ml.lightgbm_training import LightGBMGroupFoldResult


def test_lightgbm_lovo_command_parses_manifest_and_checkpoint_dir(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    args = build_parser().parse_args([
        "lightgbm-lovo", str(manifest), "--checkpoint-dir", str(tmp_path / "models")
    ])
    assert args.command == "lightgbm-lovo"
    assert args.manifest == manifest
    assert args.checkpoint_dir == tmp_path / "models"
    assert args.config == Path("config/train.yaml")


def test_lightgbm_lovo_dispatch_calls_manifest_pipeline(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("videos: []\n", encoding="utf-8")
    config_path = tmp_path / "train.yaml"
    config_path.write_text("logging:\n  level: INFO\n", encoding="utf-8")
    settings = Config({
        "data": {"manifest": str(manifest)},
        "logging": {"level": "INFO"},
        "paths": {"cache_dir": str(tmp_path / "cache")},
    })
    monkeypatch.setattr("autocut.config_loader.load_config", lambda path: settings)
    captured = {}

    expected = (
        LightGBMGroupFoldResult(
            held_out_group="ascoli",
            train_video_ids=("castelli",),
            validation_video_ids=("ascoli",),
            summary={"validation_metrics": {"accuracy": 0.75}},
        ),
    )

    def fake_pipeline(path, loaded_settings, *, checkpoint_dir=None):
        captured.update(path=path, settings=loaded_settings, checkpoint_dir=checkpoint_dir)
        return expected

    monkeypatch.setattr("autocut.pipeline.lightgbm_training.run_lightgbm_lovo_from_manifest", fake_pipeline)
    checkpoint_dir = tmp_path / "models"
    assert main([
        "lightgbm-lovo", str(manifest), "--config", str(config_path),
        "--checkpoint-dir", str(checkpoint_dir),
    ]) == 0
    assert captured == {
        "path": manifest,
        "settings": settings,
        "checkpoint_dir": checkpoint_dir,
    }
    output = capsys.readouterr().out
    assert '"held_out_group": "ascoli"' in output
    assert '"accuracy": 0.75' in output

from pathlib import Path

from autocut import cli


def test_audit_dispatch_prints_report_and_returns_success(tmp_path, monkeypatch, capsys):
    source = tmp_path / "manifest.yaml"
    source.write_text("videos: []\n", encoding="utf-8")
    config = tmp_path / "autocut.yaml"
    config.write_text("selection:\n  min_clip: 3.0\n", encoding="utf-8")
    training_config = tmp_path / "train.yaml"
    training_config.write_text("data:\n  train_groups: []\n  validation_groups: []\n")

    captured = {}

    def fake_audit(path, settings, *, training_config):
        captured["source"] = path
        captured["config"] = training_config
        return {"status": "warning", "summary": {"manifest_entries": 0}}

    monkeypatch.setattr("autocut.dataset.audit.audit_source", fake_audit)
    assert cli.main([
        "audit", str(source), "--config", str(config),
        "--training-config", str(training_config),
    ]) == 0
    assert captured["source"] == source
    assert captured["config"] == training_config
    assert '"status": "warning"' in capsys.readouterr().out


def test_audit_dispatch_returns_nonzero_for_audit_errors(tmp_path, monkeypatch):
    source = tmp_path / "annotations.txt"
    source.write_text("placeholder", encoding="utf-8")
    config = tmp_path / "autocut.yaml"
    config.write_text("selection:\n  min_clip: 3.0\n")

    monkeypatch.setattr(
        "autocut.dataset.audit.audit_source",
        lambda *args, **kwargs: {"status": "error", "errors": [{"message": "bad source"}]},
    )
    assert cli.main(["audit", str(source), "--config", str(config)]) == 1


def test_audit_dispatch_reports_missing_input(tmp_path, capsys):
    config = tmp_path / "autocut.yaml"
    config.write_text("selection:\n  min_clip: 3.0\n")
    missing = tmp_path / "missing.txt"
    assert cli.main(["audit", str(missing), "--config", str(config)]) == 2
    assert "Input file not found" in capsys.readouterr().err

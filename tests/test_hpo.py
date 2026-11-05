import json
from pathlib import Path
import numpy as np
import pytest

from autocut.config_loader import Config
from autocut.pipeline import hpo


def test_run_hpo_orchestrates_study_and_exports_json(tmp_path, monkeypatch):
    settings = Config({
        "paths": {
            "cache_dir": str(tmp_path / "cache"),
            "models_dir": str(tmp_path / "models"),
        },
        "data": {
            "manifest": str(tmp_path / "manifest.yaml"),
            "sampling_fps": 1.0,
            "processing_width": 224,
            "window_duration": 5.0,
            "window_stride": 1.0,
            "train_groups": ["group_A", "group_B", "group_C"],
            "validation_groups": ["group_B"],
        },
        "training": {
            "seed": 42,
            "lr": 1e-4,
            "weight_decay": 1e-3,
            "dropout": 0.3,
            "positive_prior": 0.1,
            "hidden_dim": 64,
            "epochs": 1,
            "batch_size": 2,
        },
        "experiment": {
            "name": "test_exp",
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "results_dir": str(tmp_path / "results"),
        },
        "hpo": {
            "study_name": "test_autocut_study",
            "n_trials": 2,
            "proxy_test_groups": ["group_A"],
            "search_space": {
                "lr": {"low": 1e-5, "high": 1e-3, "log": True},
                "weight_decay": {"low": 1e-4, "high": 1e-1, "log": True},
                "dropout": {"low": 0.2, "high": 0.6},
                "positive_prior": {"low": 0.05, "high": 0.3},
                "hidden_dim": {"choices": [32, 64, 128]},
                
            },
        },
        "logging": {"level": "INFO"},
    })

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("dummy manifest content", encoding="utf-8")

    # 2. Mock run_lovo to return a valid result structure expected by the objective
    lovo_call_count = {"count": 0}
    lovo_calls = []

    def fake_run_lovo(manifest_path, settings, **kwargs):
        lovo_call_count["count"] += 1
        lovo_calls.append({"settings": settings, **kwargs})
        return {
            "aggregation": {
                "macro_group_mean": {"macro_f1": 0.75}
            },
            "folds": [
                {
                    "selected_epochs": 1,
                    "metrics": {
                        "positive_recall_on_P": 0.70,
                        "selected_fraction_of_U": 0.20
                    }
                }
            ]
        }

    monkeypatch.setattr(hpo, "run_lovo", fake_run_lovo)
    monkeypatch.setattr(hpo, "load_manifest", lambda *args, **kwargs: ["record"])
    monkeypatch.setattr(
        hpo, "_combine_records",
        lambda records, loaded_settings: (
            np.zeros((2, 54), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        hpo, "train_fixed_epochs",
        lambda **kwargs: (None, None, {"epochs": kwargs["epochs"]}),
    )

    # 3. Redirect working directory to tmp_path to isolate DB and JSON exports
    monkeypatch.chdir(tmp_path)

    # 4. Execute run_hpo
    hpo.run_hpo(manifest_path=manifest, settings=settings)

    # 5. Assertions
    # 2 trial runs + 1 final master refit = 3 total calls to run_lovo
    assert lovo_call_count["count"] == 3
    for call in lovo_calls[:2]:
        assert call["test_groups"] == ("group_A",)
        assert call["train_groups"] == ("group_B", "group_C")
        assert call["settings"].data.train_groups == ("group_B", "group_C")
    # The final production run remains a full-manifest LOVO refit/evaluation.
    assert "test_groups" not in lovo_calls[2]
    assert "train_groups" not in lovo_calls[2]

    # Check that the JSON export file was created successfully under output/hpo/
    expected_json_path = tmp_path / "output" / "hpo" / "test_autocut_study_pareto_results.json"
    assert expected_json_path.exists()

    # Verify JSON content structure and values
    with open(expected_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["study_name"] == "test_autocut_study"
    assert payload["objectives"] == [
        {"name": "positive_recall_on_P", "direction": "maximize"},
        {"name": "selected_fraction_of_U", "direction": "minimize"},
    ]
    assert payload["pareto_front"]
    assert "production_params" in payload
    assert "production_metrics" in payload
    assert payload["production_metrics"]["positive_recall_on_P"] == 0.70
    assert payload["production_metrics"]["selected_fraction_of_U"] == 0.20


def test_run_hpo_diagnostics_reuses_stored_trial_without_optimizing(tmp_path, monkeypatch):
    class StoredTrial:
        number = 27
        state = hpo.optuna.trial.TrialState.COMPLETE
        values = (0.66, 0.63)
        params = {
            "lr": 0.007,
            "weight_decay": 0.0006,
            "dropout": 0.18,
            "hidden_dim": 128,
            "positive_prior": 0.17,
            "positive_threshold": 0.11,
        }

    class StoredStudy:
        trials = [StoredTrial()]

    settings = Config({
        "paths": {"cache_dir": str(tmp_path / "cache")},
        "data": {
            "train_groups": ["ascoli", "castelli", "mentorella", "other"],
            "validation_groups": ["ascoli"],
        },
        "training": {
            "seed": 42,
            "lr": 1e-3,
            "weight_decay": 1e-3,
            "dropout": 0.3,
            "positive_prior": 0.1,
            "positive_threshold": 0.5,
            "hidden_dim": 64,
            "epochs": 1,
            "batch_size": 2,
        },
        "hpo": {
            "study_name": "existing_study",
            "proxy_test_groups": ["ascoli", "castelli", "mentorella"],
            "search_space": {
                "lr": {"low": 1e-5, "high": 1e-2, "log": True},
                "weight_decay": {"low": 1e-5, "high": 1e-2, "log": True},
                "dropout": {"low": 0.0, "high": 0.5},
                "positive_prior": {"low": 0.05, "high": 0.3},
                "positive_threshold": {"low": 0.05, "high": 0.5},
                "hidden_dim": {"choices": [64, 128]},
            },
        },
        "experiment": {"name": "diagnostic_test"},
    })

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hpo.optuna, "load_study", lambda **kwargs: StoredStudy())
    captured = {}

    def fake_run_lovo(manifest_path, settings, **kwargs):
        captured["settings"] = settings
        captured["kwargs"] = kwargs
        output = Path(settings.debug.selection_diagnostics_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("diagnostic csv", encoding="utf-8")
        return {"folds": []}

    monkeypatch.setattr(hpo, "run_lovo", fake_run_lovo)
    report = hpo.run_hpo_diagnostics(
        tmp_path / "manifest.yaml",
        settings,
        trial_number=27,
    )

    assert report == Path("output/hpo/existing_study_trial_27_selection_diagnostics.csv")
    assert report.read_text(encoding="utf-8") == "diagnostic csv"
    assert captured["kwargs"]["test_groups"] == ("ascoli", "castelli", "mentorella")
    assert captured["kwargs"]["train_groups"] == ("other",)
    assert captured["settings"].training.positive_threshold == 0.11
    assert captured["settings"].training.positive_prior == 0.17
    assert captured["settings"].debug.selection_diagnostics_trial == 27

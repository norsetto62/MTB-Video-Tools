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
    expected_json_path = tmp_path / "output" / "hpo" / "test_autocut_study_best_params.json"
    assert expected_json_path.exists()

    # Verify JSON content structure and values
    with open(expected_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["study_name"] == "test_autocut_study"
    assert "best_value" in payload
    assert "best_params" in payload
    assert "best_metrics" in payload
    assert payload["best_metrics"]["positive_recall_on_P"] == 0.70
    assert payload["best_metrics"]["selected_fraction_of_U"] == 0.20
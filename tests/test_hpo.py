import json
from pathlib import Path
import numpy as np
import pytest

from autocut.config_loader import TrainingRunConfig, TrainingConfig, CommonSettings
from autocut.pipeline import hpo
from autocut.ml.training import TrainingConfig

def test_run_hpo_orchestrates_study_and_exports_json(tmp_path, monkeypatch):
    # 1. Construct real dataclass instances matching your exact schemas
    mock_common = CommonSettings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        output_dir=tmp_path / "output",
        log_level="INFO",
        min_clip=1.0,
        window_duration=5.0,
        window_stride=1.0,
    )
    
    mock_training = TrainingConfig(
        seed=42,
        lr=1e-4,
        weight_decay=1e-3,
        dropout=0.3,
        class_weights=(0.5, 1.0, 5.0),
        hidden_dim=64,
        epochs=1,
        batch_size=2,
    )
    
    settings = TrainingRunConfig(
        common=mock_common,
        cache_dir=tmp_path / "cache",
        manifest=tmp_path / "manifest.yaml",
        sample_fps=1.0,
        processing_width=224,
        study_name="test_autocut_study",
        n_trials=2,
        proxy_test_groups=("group_A",),
        train_groups=("group_A", "group_B", "group_C"),
        validation_groups=("group_B",),
        training=mock_training,
        experiment_name="test_exp",
        checkpoint_dir=tmp_path / "checkpoints",
        results_dir=tmp_path / "results",
    )

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("dummy manifest content", encoding="utf-8")

    # 2. Mock run_lovo to return a valid result structure expected by the objective
    lovo_call_count = {"count": 0}

    def fake_run_lovo(manifest_path, settings):
        lovo_call_count["count"] += 1
        return {
            "aggregation": {
                "macro_group_mean": {"macro_f1": 0.75}
            },
            "folds": [
                {
                    "metrics": {
                        "per_class_f1": [0.70, 0.80, 0.75]
                    }
                }
            ]
        }

    monkeypatch.setattr(hpo, "run_lovo", fake_run_lovo)

    # 3. Redirect working directory to tmp_path to isolate DB and JSON exports
    monkeypatch.chdir(tmp_path)

    # 4. Execute run_hpo
    hpo.run_hpo(manifest_path=manifest, settings=settings)

    # 5. Assertions
    # 2 trial runs + 1 final master refit = 3 total calls to run_lovo
    assert lovo_call_count["count"] == 3

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
    assert payload["best_metrics"]["macro_f1"] == 0.75
    assert payload["best_metrics"]["class_0_f1"] == 0.70
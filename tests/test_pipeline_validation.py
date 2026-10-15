from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from autocut.pipeline import validation as pipeline_validation
from autocut.config_loader import TrainingRunConfig
from autocut.ml.scaling import FeatureScaler


def _settings(tmp_path: Path):
    from autocut.config_loader import CommonSettings
    from autocut.ml.training import TrainingConfig

    return TrainingRunConfig(
        common=CommonSettings(
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
            output_dir=tmp_path / "output",
            log_level="INFO",
            min_clip=3.0,
            window_duration=4.0,
            window_stride=2.0,
        ),
        cache_dir=tmp_path / "cache",
        manifest=tmp_path / "manifest.yaml",
        sample_fps=2.0,
        processing_width=480,
        study_name='',
        n_trials=0,
        proxy_test_groups=tuple([]),
        train_groups=("train",),
        validation_groups=("val",),
        training=TrainingConfig(),
        experiment_name="test",
        checkpoint_dir=tmp_path / "models",
        results_dir=tmp_path / "results",
    )


def test_run_validation_predicts_windows_and_writes_json(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("videos: []\n", encoding="utf-8")
    video = tmp_path / "eval.mp4"
    annotation = tmp_path / "eval.txt"
    video.touch()
    annotation.write_text("eval.mp4\nStart End MTB Remarks\n00:00 00:10 1 sample\n", encoding="utf-8")
    record = SimpleNamespace(
        id="eval-1",
        group_id="eval",
        video_path=video,
        annotation_path=annotation,
    )

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, batch):
            return torch.tensor([[0.0, 2.0, 1.0]] * len(batch), dtype=torch.float32)

    checkpoint = {
        "data_config": {"flow_fps": 2.0, "window": 4.0, "stride": 2.0},
        "scaler_state": FeatureScaler(enabled=False).state_dict(),
        "feature_mask": None,
        "feature_dim": 54,
        "class_mapping": {"0": "negative", "1": "ordinary", "2": "interesting"},
    }
    monkeypatch.setattr(pipeline_validation, "load_checkpoint", lambda *a, **k: checkpoint)
    monkeypatch.setattr(pipeline_validation, "restore_model", lambda *a, **k: FakeModel())
    monkeypatch.setattr(
        pipeline_validation, "load_manifest", lambda *a, **k: (record,)
    )
    monkeypatch.setattr(
        pipeline_validation, "load_training_annotations",
        lambda path: (video, [SimpleNamespace(start=0.0, end=10.0, score=1.0)]),
    )
    monkeypatch.setattr(
        "autocut.video.probe.get_video_info",
        lambda path: SimpleNamespace(duration=10.0),
    )
    monkeypatch.setattr(
        pipeline_validation, "build_feature_sequence",
        lambda *a, **k: SimpleNamespace(features=np.zeros((12, 54), dtype=np.float32)),
    )
    monkeypatch.setattr(
        pipeline_validation, "build_windows",
        lambda *a, **k: SimpleNamespace(
            X=np.zeros((3, 8, 54), dtype=np.float32),
            timestamps=np.array([[0.0, 4.0], [2.0, 6.0], [4.0, 8.0]]),
        ),
    )

    result = pipeline_validation.run_validation(
        manifest,
        settings,
        checkpoint_path=tmp_path / "model.pt",
        group_id="eval",
    )

    assert result["total_windows"] == 3
    assert [row["predicted_class"] for row in result["records"][0]["windows"]] == [1, 1, 1]
    assert result["records"][0]["windows"][0]["start"] == 0.0
    assert Path(result["results_path"]).is_file()

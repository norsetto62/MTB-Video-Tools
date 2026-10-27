"""Tests for manifest/cache-backed LightGBM dataset preparation."""
from pathlib import Path

import numpy as np
import pytest

from autocut.data_models import Annotation, FeatureSequence, VideoInfo
from autocut.dataset.manifest import DatasetRecord
from autocut.motion.features import FEATURE_NAMES
from autocut.ml.lightgbm_training import LightGBMGroupFoldResult
from autocut.ml.model import LightGBMConfig
from autocut.pipeline import lightgbm_training as pipeline


def _record(tmp_path, record_id, group_id):
    video = tmp_path / f"{record_id}.mp4"
    annotation = tmp_path / f"{record_id}.txt"
    video.touch()
    annotation.touch()
    return DatasetRecord(
        id=record_id,
        group_id=group_id,
        video_path=video,
        annotation_path=annotation,
        manifest_path=tmp_path / "manifest.yaml",
    )


def _sequence():
    timestamps = np.arange(25, dtype=np.float64) * 0.5
    features = np.zeros((25, 54), dtype=np.float32)
    # Populate global and grid flows with finite, non-degenerate values.
    features[:, FEATURE_NAMES.index("flow_u")] = 1.0
    features[:, FEATURE_NAMES.index("flow_v")] = 0.5
    for row in range(3):
        for col in range(3):
            base = 0.2 + row * 0.4 + col * 0.15
            features[:, FEATURE_NAMES.index(f"grid_{row}_{col}_u")] = base
            features[:, FEATURE_NAMES.index(f"grid_{row}_{col}_v")] = base * 0.5
    return FeatureSequence(features=features, timestamps=timestamps)


def _annotations():
    return [
        Annotation(0.0, 3.0, 0, ""),
        Annotation(3.0, 6.0, 1, ""),
        Annotation(6.0, 9.0, 2, ""),
        Annotation(9.0, 12.0, 3, ""),
    ]


def test_prepare_manifest_data_uses_canonical_cache_and_group_ids(tmp_path, monkeypatch):
    records = (_record(tmp_path, "video_a", "ascoli"), _record(tmp_path, "video_b", "castelli"))
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("videos: []\n", encoding="utf-8")
    cache_calls = []

    monkeypatch.setattr(pipeline, "load_manifest", lambda *args, **kwargs: records)
    monkeypatch.setattr(pipeline, "load_training_annotations", lambda path: (
        records[0].video_path if Path(path) == records[0].annotation_path else records[1].video_path,
        _annotations(),
    ))
    monkeypatch.setattr(pipeline, "get_video_info", lambda path: VideoInfo(640, 480, 30.0, 12.0))

    def fake_build_feature_sequence(video_path, clip, **kwargs):
        cache_calls.append((Path(video_path), Path(kwargs["cache_dir"]), clip, kwargs))
        return _sequence()

    monkeypatch.setattr(pipeline, "build_feature_sequence", fake_build_feature_sequence)
    settings = {
        "paths": {"cache_dir": str(tmp_path / "cache")},
        "data": {
            "sampling_fps": 2.0,
            "processing_width": 480,
            "window_duration": 4.0,
            "window_stride": 2.0,
            "roi_left": 0.15,
            "roi_right": 0.85,
            "roi_bottom": 0.5,
            "roi_top": 0.1,
        },
    }

    prepared, groups, data_config = pipeline.prepare_lightgbm_manifest_data(manifest, settings)

    assert set(prepared) == {"video_a", "video_b"}
    assert groups == {"video_a": "ascoli", "video_b": "castelli"}
    assert all(item.windows.X.shape[1] == 48 for item in prepared.values())
    assert all(set(np.unique(item.targets)) == {0, 1, 2, 3} for item in prepared.values())
    assert len(cache_calls) == 2
    assert all(call[1] == tmp_path / "cache" / "features" for call in cache_calls)
    assert data_config["window"] == 4.0
    assert data_config["stride"] == 2.0


def test_manifest_preparation_rejects_unannotated_record(tmp_path, monkeypatch):
    record = DatasetRecord(
        id="video_a",
        group_id="ascoli",
        video_path=tmp_path / "video_a.mp4",
        annotation_path=None,
        manifest_path=tmp_path / "manifest.yaml",
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("videos: []\n", encoding="utf-8")
    monkeypatch.setattr(pipeline, "load_manifest", lambda *args, **kwargs: (record,))
    with pytest.raises(ValueError, match="requires MTB 0..3 annotations"):
        pipeline.prepare_lightgbm_manifest_data(manifest, {"data": {}})


def test_manifest_runner_passes_prepared_data_to_grouped_lovo(tmp_path, monkeypatch):
    prepared = {"video_a": object(), "video_b": object()}
    groups = {"video_a": "ascoli", "video_b": "castelli"}
    params = {"window": 4.0, "stride": 2.0}
    monkeypatch.setattr(
        pipeline, "prepare_lightgbm_manifest_data",
        lambda *args, **kwargs: (prepared, groups, params),
    )
    expected = (LightGBMGroupFoldResult(
        held_out_group="ascoli",
        train_video_ids=("video_b",),
        validation_video_ids=("video_a",),
        summary={"validation_metrics": {"accuracy": 1.0}},
    ),)
    calls = {}

    def fake_lovo(video_data, group_by_video, **kwargs):
        calls.update(video_data=video_data, groups=group_by_video, **kwargs)
        return expected

    monkeypatch.setattr(pipeline, "train_lightgbm_lovo", fake_lovo)
    config = LightGBMConfig(num_iterations=10, min_data_in_leaf=2)
    results = pipeline.run_lightgbm_lovo_from_manifest(
        tmp_path / "manifest.yaml",
        {"paths": {"cache_dir": str(tmp_path / "cache")}},
        config=config,
        checkpoint_dir=tmp_path / "models",
    )

    assert results == expected
    assert calls["video_data"] is prepared
    assert calls["groups"] == groups
    assert calls["config"] is config
    assert calls["data_config"]["window"] == 4.0
    assert calls["data_config"]["target_policy"].startswith("annotated-duration-majority")

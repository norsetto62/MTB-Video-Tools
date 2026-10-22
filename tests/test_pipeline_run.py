from pathlib import Path

import pytest
from types import SimpleNamespace

from autocut.config_loader import Config
from autocut.data_models import VideoInfo
from autocut.pipeline import run
from autocut.selection.timeline import Timeline
from autocut.selection.candidates_generation import CandidateGenerationConfig
from autocut.ml.model import MTBClassifier
from autocut.ml.checkpoint import CHECKPOINT_VERSION

import torch

def _run_settings(checkpoint, output_dir, cache_dir, *, padding_seconds=2.0):
    return Config({
        "paths": {
            "output_dir": output_dir,
            "cache_dir": cache_dir,
        },
        "video": {"sampling_fps": 2.0, "processing_width": 480, "encoder": "h264_nvenc"},
        "features": {"window_duration": 4.0, "window_stride": 2.0},
        "model": {"checkpoint": checkpoint, "inference_batch_size": 16},
        "selection": {
            "min_clip": 3.0,
            "max_clip_duration": 0.0,
            "positive_threshold": 0.5,
            "max_gap_to_merge": 3.0,
            "padding_seconds": padding_seconds,
        },
        "audio": {
            "sync_mode": "off",
            "max_synced_clip_duration": 0.0,
            "fade_duration": 2.0,
        },
    })

def test_merge_ranges_merges_touching_and_overlapping_intervals():
    clips = [
        SimpleNamespace(start=8.0, end=10.0),
        SimpleNamespace(start=1.0, end=3.0),
        SimpleNamespace(start=3.0, end=5.0),
        SimpleNamespace(start=4.5, end=7.0),
        SimpleNamespace(start=12.0, end=13.0),
    ]
    assert run._merge_ranges(clips) == [(1.0, 7.0), (8.0, 10.0), (12.0, 13.0)]


def test_run_pipeline_preserves_mandatory_clips_and_writes_result_manifest(
    tmp_path, monkeypatch
):
    source = tmp_path / "ride.mp4"
    source.write_bytes(b"video placeholder")
    annotations = tmp_path / "annotations.txt"
    annotations.write_text(f"{source}\n00:00:02 00:00:08 *\n", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint placeholder")
    output_dir = tmp_path / "output"

    settings = _run_settings(checkpoint, output_dir, tmp_path / "cache")
    monkeypatch.setattr(
        run, "load_annotations",
        lambda *args, **kwargs: (
            [SimpleNamespace(video_name=str(source), start=2.0, end=8.0, mandatory=True)],
            None,
        ),
    )
    monkeypatch.setattr(run, "get_video_info", lambda path: VideoInfo(1920, 1080, 30.0, 20.0))
    monkeypatch.setattr(
        run, "_extract_windows",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("mandatory intervals must not be analyzed")
        ),
    )

    def fake_render(timeline, output_path, **kwargs):
        assert isinstance(timeline, Timeline)
        assert len(timeline.clips) == 1
        assert timeline.clips[0].mandatory is True
        assert timeline.clips[0].source_start == 2.0
        assert timeline.clips[0].source_end == 8.0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered")
        return SimpleNamespace(output_path=output_path, duration=6.0)

    monkeypatch.setattr(run, "render_timeline", fake_render)
    result = run.run_pipeline(annotations, settings)

    assert result["status"] == "completed"
    assert result["mandatory_count"] == 1
    assert result["candidate_count"] == 0
    assert result["inference"] == [{
        "video": str(source.resolve()),
        "annotation_order": 0,
        "range_start": 2.0,
        "range_end": 8.0,
        "window_count": 0,
        "candidate_count": 0,
        "status": "skipped_mandatory",
    }]
    assert Path(result["result_manifest"]).is_file()
    assert result["output"]["duration"] == 6.0



def test_pipeline_preserves_annotation_order_through_binary_pu_candidate_generation(
    tmp_path, monkeypatch
):
    import numpy as np

    from autocut.ml.inference import InferenceResult

    source = tmp_path / "ride.mp4"
    source.write_bytes(b"video placeholder")
    annotations = tmp_path / "annotations.txt"
    # The later source-time interval appears first in the manifest.
    annotations.write_text(
        f"{source.name}\n00:10 00:20\n00:01 00:08\n", encoding="utf-8"
    )
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint placeholder")
    output_dir = tmp_path / "output"
    settings = _run_settings(
        checkpoint, output_dir, tmp_path / "cache", padding_seconds=0.0
    )

    monkeypatch.setattr(
        run, "get_video_info",
        lambda path: VideoInfo(1920, 1080, 30.0, 30.0),
    )
    monkeypatch.setattr(
        run, "_extract_windows",
        lambda *args, **kwargs: SimpleNamespace(X=np.zeros((2, 1, 54), dtype=np.float32)),
    )
    # The pipeline loads checkpoint metadata after inference; inference itself
    # is stubbed here so this integration test focuses on real P/U candidate
    # generation, source-time offsets, and annotation-order preservation.
    monkeypatch.setattr(run, "load_checkpoint", lambda *args, **kwargs: {\n        "data_config": {"roi": {"left": 0.15, "right": 0.85, "bottom": 0.50, "top": 0.10}}\n    })
    monkeypatch.setattr(run, "validate_checkpoint", lambda payload: payload)
    inference_calls = iter([
        InferenceResult(
            timestamps=np.asarray([[0.0, 4.0], [4.0, 8.0]]),
            probabilities=np.asarray([0.8, 0.7], dtype=np.float32),
        ),
        InferenceResult(
            timestamps=np.asarray([[0.0, 3.5], [3.5, 7.0]]),
            probabilities=np.asarray([0.8, 0.7], dtype=np.float32),
        ),
    ])
    monkeypatch.setattr(
        run, "infer_windows", lambda *args, **kwargs: next(inference_calls)
    )

    captured = {}

    def fake_render(timeline, output_path, **kwargs):
        captured["timeline"] = timeline
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered")
        return SimpleNamespace(output_path=output_path, duration=timeline.duration)

    monkeypatch.setattr(run, "render_timeline", fake_render)
    result = run.run_pipeline(annotations, settings)

    timeline = captured["timeline"]
    assert [(clip.source_start, clip.annotation_order) for clip in timeline.clips] == [
        (10.0, 2),
        (1.0, 3),
    ]
    assert [item["annotation_order"] for item in result["inference"]] == [2, 3]
    assert [item["candidate_count"] for item in result["inference"]] == [1, 1]
    assert result["candidate_count"] == 2


def test_run_config_sections_normalizes_generic_config_and_domain_settings(tmp_path):
    from autocut.audio.config import AudioConfig
    from autocut.config_loader import load_config

    checkpoint = tmp_path / "model.pt"
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        f"paths:\n  output_dir: {tmp_path / 'output'}\n  cache_dir: {tmp_path / 'cache'}\n"
        f"model:\n  checkpoint: {checkpoint}\n"
        "selection:\n  min_clip: 5.0\naudio:\n  sync_mode: beat\n",
        encoding="utf-8",
    )

    settings, candidates, audio = run._run_config_sections(load_config(config_path))

    assert isinstance(settings, Config)
    assert settings.paths.output_dir == tmp_path / "output"
    assert settings.model.checkpoint == checkpoint
    assert candidates.min_candidate_duration == 5.0
    assert isinstance(audio, AudioConfig)
    assert audio.sync_mode.value == "beat"


def test_run_config_sections_rejects_invalid_video_width():
    from autocut.config_loader import Config

    config = Config({"video": {"processing_width": 0}})
    try:
        run._run_config_sections(config)
    except ValueError as exc:
        assert "video.processing_width must be a positive integer" in str(exc)
    else:
        raise AssertionError("invalid processing width should be rejected")


def test_checkpoint_roi_must_match_run_configuration(tmp_path):
    settings, _candidate_config, _audio_config = run._run_config_sections(
        run.Config({
            "paths": {
                "data_dir": str(tmp_path), "models_dir": str(tmp_path),
                "output_dir": str(tmp_path), "cache_dir": str(tmp_path / "cache"),
            },
            "video": {"roi_left": 0.20},
            "model": {"checkpoint": str(tmp_path / "model.pt")},
        })
    )
    with pytest.raises(ValueError, match="does not match configured ROI"):
        run._validate_checkpoint_roi(
            {"data_config": {"roi": {"left": 0.15, "right": 0.85, "bottom": 0.50, "top": 0.10}}},
            settings,
        )

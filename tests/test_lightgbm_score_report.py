from pathlib import Path

import numpy as np

from autocut.config_loader import Config
from autocut.data_models import FeatureSequence, FeatureWindows, VideoInfo
from autocut.dataset.manifest import DatasetRecord
from autocut.dataset.windows import LIGHTGBM_TABULAR_FEATURE_NAMES
from autocut.pipeline import lightgbm_score_report as report


class FakeBooster:
    def predict(self, values):
        return np.asarray([0.5, -1.0, 2.0], dtype=np.float32)

    def num_feature(self):
        return len(LIGHTGBM_TABULAR_FEATURE_NAMES)


def _record(tmp_path):
    video = tmp_path / "video_a.mp4"
    annotation = tmp_path / "video_a.txt"
    video.touch()
    annotation.touch()
    return DatasetRecord(
        id="video_a",
        group_id="ascoli",
        video_path=video,
        annotation_path=annotation,
        manifest_path=tmp_path / "manifest.yaml",
    )


def test_effective_config_prefers_checkpoint_preprocessing():
    payload = {
        "data_config": {
            "sampling_fps": 1.0,
            "processing_width": 320,
            "window": 6.0,
            "stride": 3.0,
            "roi_left": 0.1,
            "roi_right": 0.9,
            "roi_bottom": 0.6,
            "roi_top": 0.2,
        }
    }
    result = report._effective_data_config(
        payload,
        {"data": {"sampling_fps": 2.0, "window_duration": 4.0}},
    )
    assert result == {
        "sampling_fps": 1.0,
        "processing_width": 320,
        "window": 6.0,
        "stride": 3.0,
        "roi_left": 0.1,
        "roi_right": 0.9,
        "roi_bottom": 0.6,
        "roi_top": 0.2,
    }


def test_report_writes_global_scores_and_per_video_range(tmp_path, monkeypatch):
    record = _record(tmp_path)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("videos: []\n", encoding="utf-8")
    checkpoint = tmp_path / "ranker.json"
    checkpoint.write_text("placeholder", encoding="utf-8")
    timestamps = np.asarray([[0.0, 4.0], [2.0, 6.0], [4.0, 8.0]])
    windows = FeatureWindows(
        X=np.zeros((3, 48), dtype=np.float32),
        timestamps=timestamps,
    )
    sequence = FeatureSequence(
        features=np.zeros((12, 54), dtype=np.float32),
        timestamps=np.arange(12, dtype=np.float64) * 0.5,
    )

    monkeypatch.setattr(report, "load_lightgbm_checkpoint", lambda path: {
        "model_class": "LightGBMRanker",
        "feature_names": list(LIGHTGBM_TABULAR_FEATURE_NAMES),
        "_booster": FakeBooster(),
        "data_config": {
            "sampling_fps": 2.0, "processing_width": 480, "window": 4.0, "stride": 2.0,
            "roi_left": 0.15, "roi_right": 0.85, "roi_bottom": 0.5, "roi_top": 0.1,
        },
    })
    monkeypatch.setattr(report, "load_manifest", lambda *args, **kwargs: (record,))
    monkeypatch.setattr(report, "load_training_annotations", lambda path: (
        record.video_path, [type("Annotation", (), {"end": 8.0})()]
    ))
    monkeypatch.setattr(report, "get_video_info", lambda path: VideoInfo(640, 480, 30.0, 8.0))
    monkeypatch.setattr(report, "build_feature_sequence", lambda *args, **kwargs: sequence)
    monkeypatch.setattr(report, "engineer_lightgbm_sequence", lambda features: np.zeros((12, 8), dtype=np.float32))
    monkeypatch.setattr(report, "build_lightgbm_windows_from_features", lambda *args, **kwargs: windows)
    monkeypatch.setattr(report, "normalize_per_video_features", lambda values: values)

    settings = Config({"paths": {"cache_dir": str(tmp_path / "cache")}})
    output = tmp_path / "scores.csv"
    result = report.run_lightgbm_score_report(
        manifest, settings, checkpoint_path=checkpoint, output_path=output
    )

    assert result["videos_scored"] == 1
    assert result["windows_scored"] == 3
    assert result["global_scores_csv"] == str(output.resolve())
    assert result["per_video_scores_csv"] == str((tmp_path / "scores_by_video.csv").resolve())
    assert result["video_ranges_csv"] == str((tmp_path / "scores_video_ranges.csv").resolve())
    import csv
    with output.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["score"]) for row in rows] == [2.0, 0.5, -1.0]
    assert [int(row["rank_in_video"]) for row in rows] == [1, 2, 3]
    assert [int(row["rank_global"]) for row in rows] == [1, 2, 3]
    assert all(float(row["video_score_min"]) == -1.0 for row in rows)
    assert all(float(row["video_score_max"]) == 2.0 for row in rows)
    assert all(float(row["video_score_range"]) == 3.0 for row in rows)



def test_cli_parser_exposes_score_report():
    from autocut.cli import build_parser

    args = build_parser().parse_args([
        "lightgbm-score-report", "manifest.yaml",
        "--checkpoint", "models/ranker.json",
        "--output", "output/scores.csv",
    ])
    assert args.command == "lightgbm-score-report"
    assert args.manifest == Path("manifest.yaml")
    assert args.checkpoint == Path("models/ranker.json")
    assert args.output == Path("output/scores.csv")

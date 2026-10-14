from pathlib import Path
from types import SimpleNamespace

from autocut.data_models import VideoInfo
from autocut.pipeline import run
from autocut.selection.timeline import Timeline
from autocut.selection.candidates_generation import CandidateGenerationConfig


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

    settings = SimpleNamespace(
        checkpoint=checkpoint,
        common=SimpleNamespace(
            min_clip=3.0,
            window_duration=4.0,
            window_stride=2.0,
            output_dir=output_dir,
        ),
        sample_fps=2.0,
        processing_width=480,
        inference_batch_size=16,
        max_clip_duration=0.0,
        candidates=CandidateGenerationConfig(),
        audio=SimpleNamespace(
            sync_mode=SimpleNamespace(value="off"),
            max_clip_duration=0.0,
            fade_duration=2.0,
        ),
        cache_dir=tmp_path / "cache",
    )
    monkeypatch.setattr(
        run, "load_annotations",
        lambda *args, **kwargs: (
            [SimpleNamespace(video_name=str(source), start=2.0, end=8.0, mandatory=True)],
            None,
        ),
    )
    monkeypatch.setattr(run, "get_video_info", lambda path: VideoInfo(1920, 1080, 30.0, 20.0))
    monkeypatch.setattr(run, "_extract_windows", lambda *args, **kwargs: None)

    def fake_render(timeline, output_path):
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
    assert Path(result["result_manifest"]).is_file()
    assert result["output"]["duration"] == 6.0



def test_pipeline_preserves_annotation_order_through_candidate_generation(
    tmp_path, monkeypatch
):
    import numpy as np

    from autocut.selection.candidates_generation import CandidateClip

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
    settings = SimpleNamespace(
        checkpoint=checkpoint,
        common=SimpleNamespace(
            min_clip=3.0,
            window_duration=4.0,
            window_stride=2.0,
            output_dir=output_dir,
        ),
        sample_fps=2.0,
        processing_width=480,
        inference_batch_size=16,
        max_clip_duration=0.0,
        candidates=CandidateGenerationConfig(padding_seconds=0.0),
        audio=SimpleNamespace(
            sync_mode=SimpleNamespace(value="off"),
            max_clip_duration=0.0,
            fade_duration=2.0,
        ),
        cache_dir=tmp_path / "cache",
    )
    monkeypatch.setattr(
        run, "get_video_info",
        lambda path: VideoInfo(1920, 1080, 30.0, 30.0),
    )
    monkeypatch.setattr(
        run, "_extract_windows",
        lambda *args, **kwargs: SimpleNamespace(X=np.zeros((1, 1, 1))),
    )
    monkeypatch.setattr(run, "infer_windows", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        run, "generate_candidates",
        lambda inference, clip_duration, config: [
            CandidateClip(
                start=0.0, end=4.0, duration=4.0,
                tier="high_action", mean_score=1.0, max_score=1.0,
            )
        ],
    )
    captured = {}

    def fake_render(timeline, output_path):
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
    assert result["candidate_count"] == 2

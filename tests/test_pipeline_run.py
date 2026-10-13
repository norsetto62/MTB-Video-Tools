from pathlib import Path
from types import SimpleNamespace

from autocut.data_models import VideoInfo
from autocut.pipeline import run
from autocut.selection.timeline import Timeline


def test_merge_ranges_merges_touching_and_overlapping_intervals():
    clips = [
        SimpleNamespace(start=8.0, end=10.0),
        SimpleNamespace(start=1.0, end=3.0),
        SimpleNamespace(start=3.0, end=5.0),
        SimpleNamespace(start=4.5, end=7.0),
        SimpleNamespace(start=12.0, end=13.0),
    ]
    assert run._merge_ranges(clips) == [(1.0, 10.0), (12.0, 13.0)]


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
        target_duration=10.0,
        max_clip_duration=0.0,
        candidates=object(),
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

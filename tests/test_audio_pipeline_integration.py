"""Phase 11 integration contracts using mocked analysis and rendering."""
import logging

from autocut.audio.analysis import AudioAnalysis
from autocut.audio.config import AudioConfig, SyncMode
from autocut.audio.pipeline import prepare_audio_timeline
from autocut.selection.timeline import Timeline, TimelineClip


def make_timeline():
    clip = TimelineClip(
        video_name="ride.mp4", source_start=1.0, source_end=3.0,
        output_start=0.0, output_end=2.0, mandatory=False,
        score=1.5, tier="high_action",
    )
    return Timeline((clip,), 10.0, 2.0, 0.0)


def test_analysis_failure_warns_and_returns_unsynchronized_timeline(caplog):
    original = make_timeline()

    def fail_analysis(*args, **kwargs):
        raise RuntimeError("librosa could not decode track")

    with caplog.at_level(logging.WARNING):
        result = prepare_audio_timeline(
            original,
            music_path="music.mp3",
            config=AudioConfig(sync_mode=SyncMode.BEAT),
            analyze=fail_analysis,
        )
    assert result == original
    assert "unsynchronized" in caplog.text.lower()


def test_pipeline_reuses_analysis_and_does_not_rerun_selection():
    original = make_timeline()
    data = AudioAnalysis(
        source_path=__import__("pathlib").Path("music.mp3"), source_hash="a" * 64, duration=12.0,
        tempo_bpm=120.0, beats=(1.0, 2.0, 3.0), measures=(1.0,),
        onsets=(1.5, 2.5), combined=(1.0, 1.5, 2.0, 2.5, 3.0),
    )
    calls = {"analysis": 0}

    def fake_analysis(*args, **kwargs):
        calls["analysis"] += 1
        return data

    result = prepare_audio_timeline(
        original, music_path="music.mp3",
        config=AudioConfig(sync_mode=SyncMode.OFF),
        analyze=fake_analysis,
    )
    assert result == original
    assert calls["analysis"] == 1

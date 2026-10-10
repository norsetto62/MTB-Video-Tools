"""Contract tests for Phase 11 audio timestamp snapping and timeline sync.

These tests intentionally precede the implementation. They define the public
behavior independently of librosa and FFmpeg.
"""
from dataclasses import FrozenInstanceError

import pytest

from autocut.audio.analysis import AudioAnalysis
from autocut.audio.config import AudioConfig, SyncMode
from autocut.editing.synchronization import snap_timestamp, synchronize_timeline
from autocut.selection.timeline import Timeline, TimelineClip


def analysis(*, beats=(1.0, 2.0, 3.0), measures=(1.0, 3.0),
             onsets=(0.8, 2.2), duration=5.0):
    combined = tuple(sorted(set(round(x, 3) for x in (*beats, *onsets)
                                if 0 <= x <= duration)))
    return AudioAnalysis(
        source_path=__import__("pathlib").Path("music.mp3"),
        source_hash="a" * 64,
        duration=duration,
        tempo_bpm=120.0,
        beats=tuple(beats),
        measures=tuple(measures),
        onsets=tuple(onsets),
        combined=combined,
    )


def clip(name="ride.mp4", start=0.9, end=2.1, out_start=0.0,
         mandatory=False, score=1.0):
    return TimelineClip(
        video_name=name,
        source_start=start,
        source_end=end,
        output_start=out_start,
        output_end=out_start + end - start,
        mandatory=mandatory,
        score=None if mandatory else score,
        tier=None if mandatory else "high_action",
    )


def timeline(clips, target=10.0):
    cursor = 0.0
    placed = []
    for item in clips:
        duration = item.source_end - item.source_start
        placed.append(TimelineClip(
            video_name=item.video_name,
            source_start=item.source_start,
            source_end=item.source_end,
            output_start=cursor,
            output_end=cursor + duration,
            mandatory=item.mandatory,
            score=item.score,
            tier=item.tier,
        ))
        cursor += duration
    return Timeline(tuple(placed), target, cursor, max(0.0, cursor - target))


@pytest.mark.parametrize(
    ("mode", "timestamp", "expected"),
    [
        (SyncMode.BEAT, 1.4, 1.0),       # nearest beat
        (SyncMode.BEAT, 1.6, 2.0),
        (SyncMode.MEASURE, 2.2, 3.0),
        (SyncMode.ONSET, 2.0, 2.2),
        (SyncMode.COMBINED, 2.0, 2.0),   # exact beat wins
        (SyncMode.FORWARD_BEAT, 1.01, 2.0),
        (SyncMode.FORWARD_BEAT, 3.1, 3.1),  # no later beat => unchanged
        (SyncMode.OFF, 1.234, 1.234),
    ],
)
def test_snap_timestamp_modes(mode, timestamp, expected):
    assert snap_timestamp(timestamp, analysis(), mode=mode) == pytest.approx(expected)


def test_forward_beat_uses_first_beat_at_or_after_timestamp():
    data = analysis(beats=(1.0, 1.5, 2.0))
    assert snap_timestamp(1.5, data, mode=SyncMode.FORWARD_BEAT) == 1.5
    assert snap_timestamp(1.5001, data, mode=SyncMode.FORWARD_BEAT) == 2.0


def test_invalid_mode_is_rejected_instead_of_defaulting_to_beat():
    with pytest.raises((ValueError, TypeError)):
        snap_timestamp(1.2, analysis(), mode="nearest-ish")


@pytest.mark.parametrize("timestamp", [-0.1, float("nan"), float("inf")])
def test_invalid_timestamp_is_rejected(timestamp):
    with pytest.raises(ValueError):
        snap_timestamp(timestamp, analysis(), mode=SyncMode.BEAT)


def test_sync_adjusts_both_optional_boundaries_and_rebuilds_output_positions():
    original = timeline([
        clip(start=0.9, end=2.1, out_start=100),
        clip(name="other.mp4", start=2.9, end=4.1, out_start=200),
    ])
    result = synchronize_timeline(
        original, analysis(beats=(1.0, 2.0, 3.0, 4.0)),
        config=AudioConfig(sync_mode=SyncMode.BEAT, max_clip_duration=0),
    )
    assert [(c.source_start, c.source_end) for c in result.clips] == [
        (1.0, 2.0), (3.0, 4.0)
    ]
    assert [(c.output_start, c.output_end) for c in result.clips] == [
        (0.0, 1.0), (1.0, 2.0)
    ]
    assert result.duration == pytest.approx(2.0)
    assert result is not original


def test_mandatory_clip_is_unchanged_even_when_mode_is_enabled():
    mandatory = clip(start=0.9, end=2.1, mandatory=True)
    original = timeline([mandatory])
    result = synchronize_timeline(
        original, analysis(),
        config=AudioConfig(sync_mode=SyncMode.BEAT, max_clip_duration=1.0),
    )
    assert result.clips[0].source_start == 0.9
    assert result.clips[0].source_end == 2.1
    assert result.clips[0].mandatory is True


def test_unavailable_analysis_falls_back_to_unchanged_timeline():
    original = timeline([clip()])
    result = synchronize_timeline(
        original, None,
        config=AudioConfig(sync_mode=SyncMode.BEAT, max_clip_duration=0),
    )
    assert result == original


def test_off_mode_preserves_timeline_exactly():
    original = timeline([clip()])
    result = synchronize_timeline(
        original, analysis(),
        config=AudioConfig(sync_mode=SyncMode.OFF, max_clip_duration=0),
    )
    assert result == original


def test_sync_does_not_exceed_positive_maximum_clip_duration():
    original = timeline([clip(start=0.9, end=2.1)])
    result = synchronize_timeline(
        original, analysis(beats=(0.0, 1.0, 2.0, 3.0)),
        config=AudioConfig(sync_mode=SyncMode.BEAT, max_clip_duration=1.0),
    )
    assert result.clips[0].duration <= 1.0 + 1e-9


def test_sync_keeps_original_optional_interval_if_snapping_breaks_budget():
    original = timeline([clip(start=0.9, end=2.1)], target=1.2)
    result = synchronize_timeline(
        original, analysis(beats=(0.0, 2.0, 3.0)),
        config=AudioConfig(sync_mode=SyncMode.BEAT, max_clip_duration=0),
    )
    assert (result.clips[0].source_start, result.clips[0].source_end) == (0.9, 2.1)


def test_sync_rejects_invalid_config_values():
    with pytest.raises((ValueError, TypeError)):
        AudioConfig(max_clip_duration=-1)
    with pytest.raises((ValueError, TypeError)):
        AudioConfig(fade_duration=-0.1)

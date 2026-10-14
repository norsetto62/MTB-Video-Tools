"""Pure music-event timestamp snapping and immutable timeline synchronization."""

from __future__ import annotations

import math
from numbers import Real

from autocut.audio.analysis import AudioAnalysis
from autocut.audio.config import AudioConfig, SyncMode
from autocut.selection.timeline import Timeline, TimelineClip


def _timestamp(value: Real) -> float:
    """Validate and normalize a timestamp without accepting booleans or NaN."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("timestamp must be a finite real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("timestamp must be finite.")
    if result < 0:
        raise ValueError("timestamp must be non-negative.")
    return result


def _sync_mode(value: SyncMode | str) -> SyncMode:
    if isinstance(value, SyncMode):
        return value
    if isinstance(value, str):
        try:
            return SyncMode(value)
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in SyncMode)
            raise ValueError(
                f"mode must be one of: {choices}; got {value!r}."
            ) from exc
    raise TypeError("mode must be a SyncMode or a valid mode string.")


def snap_timestamp(
    timestamp: float,
    analysis: AudioAnalysis,
    *,
    mode: SyncMode,
) -> float:
    """Snap a timestamp to an analysis event grid.

    Nearest-event modes choose the closest timestamp; ties resolve to the
    earlier event because validated grids are sorted. Forward-beat chooses
    the first beat at or after the timestamp. Empty grids leave it unchanged.
    """
    value = _timestamp(timestamp)
    selected_mode = _sync_mode(mode)
    if not isinstance(analysis, AudioAnalysis):
        raise TypeError("analysis must be an AudioAnalysis instance.")

    if selected_mode is SyncMode.OFF:
        return value

    if selected_mode is SyncMode.BEAT:
        grid = analysis.beats
    elif selected_mode is SyncMode.MEASURE:
        grid = analysis.measures
    elif selected_mode is SyncMode.FORWARD_BEAT:
        for beat in analysis.beats:
            if beat >= value:
                return beat
        return value
    elif selected_mode is SyncMode.ONSET:
        grid = analysis.onsets
    else:
        grid = analysis.combined

    if not grid:
        return value
    return min(grid, key=lambda event: abs(event - value))


def _rebuild_timeline(
    timeline: Timeline,
    intervals: tuple[tuple[float, float], ...],
) -> Timeline:
    """Recompute output positions and total duration from source spans."""
    clips: list[TimelineClip] = []
    cursor = 0.0
    for clip, (start, end) in zip(timeline.clips, intervals, strict=True):
        duration = end - start
        clips.append(TimelineClip(
            video_name=clip.video_name,
            source_start=start,
            source_end=end,
            output_start=cursor,
            output_end=cursor + duration,
            mandatory=clip.mandatory,
            score=clip.score,
            tier=clip.tier,
            annotation_order=clip.annotation_order,
        ))
        cursor += duration

    return Timeline(clips=tuple(clips), duration=cursor)


def synchronize_timeline(
    timeline: Timeline,
    analysis: AudioAnalysis | None,
    *,
    config: AudioConfig,
) -> Timeline:
    """Return a synchronized timeline without mutating the original.

    Mandatory clip source intervals are immutable. Optional intervals are
    snapped at both boundaries, then checked for positive duration and the
    configured per-clip maximum. There is no global output-duration budget.
    A missing analysis or disabled mode is a no-op; analysis-failure logging
    belongs to the calling pipeline so it can warn only once.
    """
    if not isinstance(timeline, Timeline):
        raise TypeError("timeline must be a Timeline instance.")
    if not isinstance(config, AudioConfig):
        raise TypeError("config must be an AudioConfig instance.")
    if analysis is not None and not isinstance(analysis, AudioAnalysis):
        raise TypeError("analysis must be an AudioAnalysis instance or None.")

    if analysis is None or config.sync_mode is SyncMode.OFF or not timeline.clips:
        return timeline

    original_intervals = tuple(
        (clip.source_start, clip.source_end) for clip in timeline.clips
    )
    proposed: list[tuple[float, float]] = []

    for clip, (original_start, original_end) in zip(
        timeline.clips, original_intervals, strict=True
    ):
        if clip.mandatory:
            proposed.append((original_start, original_end))
            continue

        start = snap_timestamp(original_start, analysis, mode=config.sync_mode)
        end = snap_timestamp(original_end, analysis, mode=config.sync_mode)

        # Do not accept reversed/empty intervals or intervals that violate
        # the Phase 10 maximum-duration contract. Trimming itself belongs to
        # selection, where inference-window evidence is available.
        if end <= start:
            proposed.append((original_start, original_end))
            continue
        duration = end - start
        if (
            config.max_clip_duration > 0
            and duration > config.max_clip_duration + 1e-9
        ):
            proposed.append((original_start, original_end))
            continue
        proposed.append((start, end))

    proposed_intervals = tuple(proposed)

    if proposed_intervals == original_intervals:
        return timeline
    return _rebuild_timeline(timeline, proposed_intervals)

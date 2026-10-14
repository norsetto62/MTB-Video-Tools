"""Prepare an existing selection timeline for optional music synchronization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from autocut.audio.analysis import AudioAnalysis, analyze_audio
from autocut.audio.config import AudioConfig
from autocut.editing.synchronization import synchronize_timeline
from autocut.selection.timeline import Timeline

logger = logging.getLogger(__name__)

AudioAnalyzer = Callable[..., AudioAnalysis]


def prepare_audio_timeline(
    timeline: Timeline,
    music_path: str | Path,
    *,
    config: AudioConfig,
    analyze: AudioAnalyzer | None = None,
    cache_dir: str | Path | None = None,
    analysis: AudioAnalysis | None = None,
) -> Timeline:
    """Analyze music and synchronize an already-selected timeline.

    This function deliberately does not run candidate generation, scoring, or
    selection. It receives a finalized timeline and returns either its
    synchronized replacement or the original timeline if audio analysis
    fails. Analysis is attempted even when synchronization is disabled so
    callers can reuse/cache analysis for other audio-related work.

    Args:
        timeline: Timeline produced by the selection stage.
        music_path: Music file to analyze.
        config: Audio synchronization settings.
        analyze: Optional injectable analyzer, useful for tests or custom
            analysis implementations. Defaults to :func:`analyze_audio`.
        cache_dir: Optional directory for the audio analysis cache.

    Returns:
        The synchronized timeline, or the original timeline if analysis
        raises an exception.
    """
    if not isinstance(timeline, Timeline):
        raise TypeError("timeline must be a Timeline instance.")
    if not isinstance(config, AudioConfig):
        raise TypeError("config must be an AudioConfig instance.")
    if not isinstance(music_path, (str, Path)):
        raise TypeError("music_path must be a string or Path.")
    if analyze is not None and not callable(analyze):
        raise TypeError("analyze must be callable or None.")
    if cache_dir is not None and not isinstance(cache_dir, (str, Path)):
        raise TypeError("cache_dir must be a string or Path, or None.")

    if analysis is not None and not isinstance(analysis, AudioAnalysis):
        raise TypeError("analysis must be an AudioAnalysis instance or None.")
    analyzer = analyze_audio if analyze is None else analyze
    try:
        if analysis is None:
            analysis = analyzer(music_path, cache_dir=cache_dir)
        if not isinstance(analysis, AudioAnalysis):
            raise TypeError("Audio analyzer must return an AudioAnalysis instance.")
    except Exception as exc:
        logger.warning(
            "Audio analysis failed; returning the original unsynchronized "
            "timeline: %s",
            exc,
        )
        return timeline

    return synchronize_timeline(timeline, analysis, config=config)

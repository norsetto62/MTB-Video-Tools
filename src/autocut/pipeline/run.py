"""End-to-end production orchestration for AutoCut."""

from __future__ import annotations

import json
import math
import logging
from pathlib import Path
import time
from typing import Iterable, Mapping

from ..config_loader import Config
from ..audio.config import AudioConfig

import numpy as np

from ..annotations import load_annotations
from ..data_models import Clip, VideoInfo
from ..dataset.cache_paths import feature_cache_dir, audio_cache_dir
from ..editing.audio import render_timeline, render_with_music
from ..selection.selection import MandatoryClip, SelectionCandidate, select_candidates
from ..selection.candidate_types import CandidateClip, WindowEvidence
from ..selection.timeline import Timeline, assemble_timeline
from ..video.probe import get_video_info
from ..audio.pipeline import prepare_audio_timeline
from ..audio.analysis import AudioAnalysis, analyze_audio

logger = logging.getLogger(__name__)



def _merge_ranges(clips: Iterable[object]) -> list[tuple[float, float]]:
    """Return sorted, merged processing ranges for a video's requested clips."""
    intervals = sorted((float(item.start), float(item.end)) for item in clips)
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
    return merged


def _run_config_sections(config: Config) -> tuple[Config, AudioConfig]:
    """Validate run-specific settings and construct only domain-specific configs."""
    if not isinstance(config, Config):
        raise TypeError("run_pipeline expects a generic Config loaded by load_config().")

    raw = config.to_dict()
    allowed_sections = {"paths", "video", "features", "model", "selection", "audio", "logging"}
    unknown_sections = set(raw) - allowed_sections
    if unknown_sections:
        raise ValueError(f"Unknown top-level configuration section(s): {', '.join(sorted(unknown_sections))}.")

    defaults = {
        "paths": {"data_dir": "data", "models_dir": "models", "output_dir": "output", "cache_dir": "data/cache"},
        "video": {"sampling_fps": 2.0, "processing_width": 480, "encoder": "h264_nvenc", "resolution": "2.7k",
                   "roi_left": 0.15, "roi_right": 0.85, "roi_bottom": 0.50, "roi_top": 0.10},
        "features": {"window_duration": 4.0, "window_stride": 2.0},
        "model": {"checkpoint": "models/lightgbm/ranker.json"},
        "selection": {"min_clip": 3.0, "max_clip_duration": 0.0},
        "audio": {"sync_mode": "off", "max_synced_clip_duration": 0.0, "fade_duration": 2.0},
        "logging": {"level": "INFO"},
    }
    allowed_keys = {
        "paths": {"data_dir", "models_dir", "output_dir", "cache_dir"},
        "video": {"sampling_fps", "processing_width", "encoder", "resolution", "roi_left", "roi_right", "roi_bottom", "roi_top"},
        "features": {"window_duration", "window_stride"},
        "model": {"checkpoint"},
        "selection": set(defaults["selection"]),
        "audio": {"sync_mode", "max_synced_clip_duration", "fade_duration"},
        "logging": {"level"},
    }
    for name, values in defaults.items():
        section = raw.setdefault(name, {})
        if not isinstance(section, Mapping):
            raise ValueError(f"Configuration section {name!r} must be a mapping.")
        unknown = set(section) - allowed_keys[name]
        if unknown:
            raise ValueError(f"Unknown field(s) in {name!r}: {', '.join(sorted(unknown))}.")
        for key, value in values.items():
            section.setdefault(key, value)

    def path_value(section: str, key: str) -> Path:
        value = raw[section][key]
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"{section}.{key} must be a non-empty path string.")
        return Path(value).expanduser()

    def number(section: str, key: str, *, minimum: float, exclusive: bool = False) -> float:
        value = raw[section][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{section}.{key} must be a finite number.")
        result = float(value)
        if not math.isfinite(result) or result < minimum or (exclusive and result == minimum):
            relation = "greater than" if exclusive else "at least"
            raise ValueError(f"{section}.{key} must be finite and {relation} {minimum}.")
        return result

    for key in allowed_keys["paths"]:
        raw["paths"][key] = path_value("paths", key)
    raw["model"]["checkpoint"] = path_value("model", "checkpoint")
    for section, key in (
        ("video", "sampling_fps"), ("features", "window_duration"),
        ("features", "window_stride"), ("selection", "min_clip"),
    ):
        raw[section][key] = number(section, key, minimum=0.0, exclusive=True)
    raw["selection"]["max_clip_duration"] = number(
        "selection", "max_clip_duration", minimum=0.0
    )

    from ..dataset.sequence import validate_roi
    roi = validate_roi(raw["video"]["roi_left"], raw["video"]["roi_right"], raw["video"]["roi_bottom"], raw["video"]["roi_top"])
    for key, value in zip(("roi_left", "roi_right", "roi_bottom", "roi_top"), roi):
        raw["video"][key] = value

    width = raw["video"]["processing_width"]
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("video.processing_width must be a positive integer.")
    batch_size = raw["model"]["inference_batch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("model.inference_batch_size must be a positive integer.")
    if raw["video"]["encoder"] not in {"h264_nvenc", "libx264"}:
        raise ValueError("video.encoder must be h264_nvenc or libx264.")
    if raw["video"]["resolution"] not in {"4k", "2.7k", "1080p"}:
        raise ValueError("video.resolution must be one of 4k, 2.7k, or 1080p.")
    level = raw["logging"]["level"]
    if not isinstance(level, str) or level.upper() not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}:
        raise ValueError("logging.level must be a standard Python logging level.")

    audio_config = AudioConfig(**raw["audio"])
    return Config(raw), audio_config

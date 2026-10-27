"""End-to-end production orchestration for AutoCut."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import logging
from pathlib import Path
import time
from typing import Iterable, Mapping

from ..config_loader import Config
from ..audio.config import AudioConfig
from ..selection.candidates_generation import CandidateGenerationConfig

import numpy as np

from ..annotations import load_annotations
from ..data_models import Clip, FeatureSequence, FeatureWindows, VideoInfo
from ..dataset.windows import build_windows
from ..dataset.cache_paths import feature_cache_dir, audio_cache_dir
from ..editing.audio import render_timeline, render_with_music
from ..ml.inference import infer_windows, InferenceResult
from ..motion.engineered import MODEL_FEATURE_NAMES, engineer_motion_sequence
from ..ml.checkpoint import load_checkpoint, validate_checkpoint
from ..selection.selection import MandatoryClip, SelectionCandidate, select_candidates
from ..selection.candidates_generation import CandidateClip, generate_candidates
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


def _extract_windows(
    video_path: Path,
    start: float,
    end: float,
    settings: Config,
    *,
    feature_names: list[str] | tuple[str, ...] | None = None,
) -> FeatureWindows | None:
    """Use the canonical dataset sequence builder, then construct windows."""
    from ..dataset.sequence import build_feature_sequence

    cache_clip = Clip(
        video_name=str(video_path),
        start=float(start),
        end=float(end),
        mandatory=False,
    )
    try:
        sequence = build_feature_sequence(
            video_path,
            cache_clip,
            sample_fps=settings.video.sampling_fps,
            processing_width=settings.video.processing_width,
            cache_dir=feature_cache_dir(settings),
            roi_left=settings.video.roi_left,
            roi_right=settings.video.roi_right,
            roi_bottom=settings.video.roi_bottom,
            roi_top=settings.video.roi_top,
        )
        # Dataset sequence timestamps are absolute; candidate windows are
        # local to this processing range and translated back by the caller.
        sequence = FeatureSequence(
            features=sequence.features,
            timestamps=np.asarray(sequence.timestamps - start, dtype=np.float64),
        )
        # Seven-signal checkpoints require the transform over the continuous
        # sequence before overlapping windows are formed (especially Savitzky-Golay).
        if feature_names is not None and list(feature_names) == list(MODEL_FEATURE_NAMES):
            transformed = engineer_motion_sequence(sequence.features)
            from ..dataset.windows import build_windows_from_features

            return build_windows_from_features(
                transformed,
                sequence.timestamps,
                window=settings.features.window_duration,
                stride=settings.features.window_stride,
            )
        return build_windows(
            sequence,
            window=settings.features.window_duration,
            stride=settings.features.window_stride,
        )
    except ValueError as exc:
        if end - start < settings.features.window_duration:
            logger.info(
                "Skipping candidate inference for short range %s [%.3f, %.3f): %s",
                video_path, start, end, exc,
            )
            return None
        raise

def _offset_candidate(candidate: CandidateClip, offset: float) -> CandidateClip:
    """Translate a range-local candidate and its evidence to source timestamps."""
    from ..selection.candidates_generation import WindowEvidence

    return CandidateClip(
        start=candidate.start + offset,
        end=candidate.end + offset,
        duration=candidate.duration,
        mean_probability=candidate.mean_probability,
        max_probability=candidate.max_probability,
        window_evidence=tuple(
            WindowEvidence(
                start=window.start + offset,
                end=window.end + offset,
                probability=window.probability,
            )
            for window in candidate.window_evidence
        ),
    )


def _run_config_sections(config: Config) -> tuple[Config, CandidateGenerationConfig, AudioConfig]:
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
        "video": {"sampling_fps": 2.0, "processing_width": 480, "encoder": "h264_nvenc", "roi_left": 0.15, "roi_right": 0.85, "roi_bottom": 0.50, "roi_top": 0.10},
        "features": {"window_duration": 4.0, "window_stride": 2.0},
        "model": {"checkpoint": "models/best_model.pt", "inference_batch_size": 256},
        "selection": {
            "min_clip": 3.0, "max_clip_duration": 0.0,
            "min_candidate_duration": 3.0, "max_gap_to_merge": 3.0, "padding_seconds": 2.0,
            "positive_threshold": 0.5,
        },
        "audio": {"sync_mode": "off", "max_synced_clip_duration": 0.0, "fade_duration": 2.0},
        "logging": {"level": "INFO"},
    }
    allowed_keys = {
        "paths": {"data_dir", "models_dir", "output_dir", "cache_dir"},
        "video": {"sampling_fps", "processing_width", "encoder", "roi_left", "roi_right", "roi_bottom", "roi_top"},
        "features": {"window_duration", "window_stride"},
        "model": {"checkpoint", "inference_batch_size"},
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
    level = raw["logging"]["level"]
    if not isinstance(level, str) or level.upper() not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}:
        raise ValueError("logging.level must be a standard Python logging level.")

    candidate_fields = {
        "min_candidate_duration", "max_gap_to_merge", "padding_seconds", "positive_threshold",
    }
    candidate_values = {
        key: value for key, value in raw["selection"].items() if key in candidate_fields
    }
    
    # min_clip is the sole final candidate-duration filter.
    candidate_values["min_candidate_duration"] = raw["selection"]["min_clip"]
    candidate_config = CandidateGenerationConfig(**candidate_values)
    audio_config = AudioConfig(**raw["audio"])
    return Config(raw), candidate_config, audio_config


def _config_summary(
    settings: Config,
    candidate_config: CandidateGenerationConfig,
    audio_config: AudioConfig,
) -> dict:
    """Build a stable, JSON-safe subset of the effective run configuration."""
    return {
        "sample_fps": settings.video.sampling_fps,
        "processing_width": settings.video.processing_width,
        "video_encoder": settings.video.encoder,
        "window_duration": settings.features.window_duration,
        "window_stride": settings.features.window_stride,
        "checkpoint": str(settings.model.checkpoint),
        "inference_batch_size": settings.model.inference_batch_size,
        "min_clip": settings.selection.min_clip,
        "max_clip_duration": settings.selection.max_clip_duration,
        "candidate_generation": asdict(candidate_config),
        "audio": {
            "sync_mode": audio_config.sync_mode.value,
            "max_synced_clip_duration": audio_config.max_synced_clip_duration,
            "fade_duration": audio_config.fade_duration,
        },
    }

def run_pipeline(
    annotation_path: str | Path,
    settings: Config,
    *,
    output_path: str | Path | None = None,
    video_encoder: str | None = None,
) -> dict:
    """Run AutoCut from an annotation manifest through final rendering.

    The annotation intervals define the source ranges to analyze. Mandatory
    intervals are separately protected during selection and are never trimmed
    or synchronized. The returned dictionary is also written as a result JSON.
    """
    settings, candidate_config, audio_config = _run_config_sections(settings)
    annotation = Path(annotation_path)
    if not annotation.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation}")
    if not settings.model.checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {settings.model.checkpoint}")

    destination = (
        Path(output_path)
        if output_path is not None
        else settings.paths.output_dir / f"{annotation.stem}_autocut.mp4"
    )
    logger.info("AutoCut run starting.")
    logger.info("Annotation manifest: %s", annotation.resolve())
    logger.info("Model checkpoint: %s", settings.model.checkpoint.resolve())
    logger.info("Planned output: %s", destination.resolve())

    clips, music_directive = load_annotations(
        annotation,
        min_clip=settings.selection.min_clip,
    )
    if not clips:
        raise ValueError(
            f"No processing intervals were found in annotation manifest: {annotation}"
        )

    logger.info("Loaded %d processing interval(s).", len(clips))

    # Establish stable video ordering from first appearance in the manifest.
    video_order: dict[str, int] = {}
    clips_by_video: dict[str, list] = {}
    for clip in clips:
        video_name = str(Path(clip.video_name).resolve())
        if video_name not in video_order:
            video_order[video_name] = len(video_order)
            clips_by_video[video_name] = []
        clips_by_video[video_name].append(clip)

    video_metadata: dict[str, VideoInfo] = {}
    for video_name, video_clips in clips_by_video.items():
        info = get_video_info(Path(video_name))
        ranges = _merge_ranges(video_clips)
        logger.info(
            "Source %s: %.1fs, %dx%d at %.3f FPS; %d processing range(s).",
            video_name, info.duration, info.width, info.height, info.fps, len(ranges),
        )
        video_metadata[video_name] = info
        for clip in video_clips:
            if clip.start < 0 or clip.end > info.duration + 1e-6:
                raise ValueError(
                    f"Annotation interval [{clip.start:.3f}, {clip.end:.3f}) "
                    f"exceeds video duration {info.duration:.3f}s: {video_name}"
                )

    if music_directive is not None and not Path(music_directive.path).is_file():
        raise FileNotFoundError(f"Background music not found: {music_directive.path}")

    audio_analysis: AudioAnalysis | None = None
    if music_directive is not None:
        try:
            audio_analysis = analyze_audio(
                music_directive.path,
                cache_dir=audio_cache_dir(settings),
            )
        except Exception as exc:
            logger.warning(
                "Audio analysis failed before selection; duration-aware selection "
                "is unavailable: %s",
                exc,
            )

    if not clips_by_video:
        raise ValueError("The annotation manifest does not identify any video ranges.")

    all_candidates: list[SelectionCandidate] = []
    inference_summaries: list[dict] = []
    logger.info("Starting motion analysis and model inference.")

    # Process each annotation interval independently. This retains the
    # manifest row identity throughout inference and candidate generation;
    # merging source ranges here would erase which interval produced a clip.
    for fallback_order, clip in enumerate(clips):
        source_annotation_order = getattr(clip, "annotation_order", None)
        annotation_order = (
            source_annotation_order
            if source_annotation_order is not None
            else fallback_order
        )
        video_name = str(Path(clip.video_name).resolve())
        start, end = float(clip.start), float(clip.end)
        if clip.mandatory:
            logger.info(
                "Skipping motion analysis and inference for mandatory interval "
                "%s [%.3f, %.3f) (annotation #%d).",
                Path(video_name).name, start, end, annotation_order,
            )
            inference_summaries.append({
                "video": video_name,
                "annotation_order": annotation_order,
                "range_start": start,
                "range_end": end,
                "window_count": 0,
                "candidate_count": 0,
                "status": "skipped_mandatory",
            })
            continue
        order = video_order[video_name]
        if isinstance(settings.model.checkpoint, (str, Path)):
            payload = load_checkpoint(settings.model.checkpoint, map_location="cpu")
        elif isinstance(settings.model.checkpoint, Mapping):
            payload = validate_checkpoint(dict(settings.model.checkpoint))
        else:
            raise TypeError("checkpoint must be a path or a checkpoint mapping.")
        data_config = payload.get("data_config", {})
        checkpoint_feature_names = payload.get("feature_names")
        trained_roi = data_config.get("roi") if isinstance(data_config, Mapping) else None
        if not isinstance(trained_roi, Mapping):
            raise ValueError(
                "Checkpoint has no ROI metadata. Retrain the model with the current "
                "ROI-aware feature pipeline before inference."
            )
        settings = settings.with_overrides({
            "data.roi_left": trained_roi["left"],
            "data.roi_right": trained_roi["right"],
            "data.roi_bottom": trained_roi["bottom"],
            "data.roi_top": trained_roi["top"],
        })
        windows = _extract_windows(
            Path(video_name), start, end, settings,
            feature_names=checkpoint_feature_names,
        )
        if windows is None:
            inference_summaries.append({
                "video": video_name,
                "annotation_order": annotation_order,
                "range_start": start,
                "range_end": end,
                "window_count": 0,
                "candidate_count": 0,
                "status": "skipped_insufficient_features",
            })
            continue
        logger.info(
            "Running model inference on %d window(s) from %s [%.1f, %.1f]s "
            "(annotation #%d).",
            int(windows.X.shape[0]), Path(video_name).name, start, end,
            annotation_order,
        )
        inference = infer_windows(
            settings.model.checkpoint,
            windows,
            batch_size=settings.model.inference_batch_size,
        )
        cleaned_inference = InferenceResult(
            timestamps=inference.timestamps,
            probabilities=inference.probabilities,
        )
        candidates = generate_candidates(
            cleaned_inference,
            clip_duration=end - start,
            config=candidate_config,
        )
        logger.info(
            "Annotation #%d result: %d inference window(s), %d candidate clip(s).",
            annotation_order, int(windows.X.shape[0]), len(candidates),
        )
        inference_summaries.append({
            "video": video_name,
            "annotation_order": annotation_order,
            "range_start": start,
            "range_end": end,
            "window_count": int(windows.X.shape[0]),
            "candidate_count": len(candidates),
            "status": "processed",
        })
        all_candidates.extend(
            SelectionCandidate(
                video_name=video_name,
                video_order=order,
                candidate=_offset_candidate(candidate, start),
                annotation_order=annotation_order,
            )
            for candidate in candidates
        )

    mandatory = tuple(
        MandatoryClip(
            video_name=str(Path(clip.video_name).resolve()),
            start=float(clip.start),
            end=float(clip.end),
            video_order=video_order[str(Path(clip.video_name).resolve())],
            sequence_order=(
                getattr(clip, "annotation_order", None)
                if getattr(clip, "annotation_order", None) is not None
                else index
            ),
        )
        for index, clip in enumerate(clips)
        if clip.mandatory
    )
    logger.info("Selecting from %d candidate(s); soundtrack duration=%s.", len(all_candidates), f"{audio_analysis.duration:.3f}s" if audio_analysis else "unavailable")
    selection = select_candidates(
        all_candidates,
        mandatory,
        min_clip_duration=settings.selection.min_clip,
        max_clip_duration=settings.selection.max_clip_duration,
        max_total_duration=audio_analysis.duration if audio_analysis is not None else None,
    )
    timeline = assemble_timeline(selection, mandatory)

    logger.info(
        "Selection complete: %d optional clip(s) selected, %d mandatory clip(s); "
        "timeline contains %d clip(s).",
        len(selection.selected), len(mandatory), len(timeline.clips),
    )

    if not timeline.clips:
        raise RuntimeError(
            "No highlight candidates were accepted and the manifest contains "
            "no mandatory clips; no output video can be rendered. Review the "
            "model checkpoint or candidate thresholds."
        )

    if music_directive is not None:
        timeline = prepare_audio_timeline(
            timeline,
            music_directive.path,
            config=audio_config,
            cache_dir=settings.paths.cache_dir / "audio",
            analysis=audio_analysis,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Rendering %d timeline clip(s) to %s.", len(timeline.clips), destination.resolve())
    if music_directive is None:
        render_result = render_timeline(
            timeline, destination,
            video_encoder=video_encoder or settings.video.encoder,
        )
    else:
        render_result = render_with_music(
            timeline,
            music_directive.path,
            destination,
            config=audio_config,
            mix_original_audio=music_directive.mix,
            video_encoder=(video_encoder or settings.video.encoder),
        )

    result_path = destination.with_name(f"{destination.stem}_result.json")
    result = {
        "schema_version": 1,
        "status": "completed",
        "annotation_manifest": str(annotation.resolve()),
        "checkpoint": str(settings.model.checkpoint.resolve()),
        "configuration": _config_summary(settings, candidate_config, audio_config),
        "sources": [
            {
                "path": name,
                "duration": video_metadata[name].duration,
                "width": video_metadata[name].width,
                "height": video_metadata[name].height,
                "fps": video_metadata[name].fps,
                "size_bytes": Path(name).stat().st_size,
                "mtime_ns": Path(name).stat().st_mtime_ns,
            }
            for name in clips_by_video
        ],
        "processing_ranges": [
            {"video": name, "start": start, "end": end}
            for name, video_clips in clips_by_video.items()
            for start, end in _merge_ranges(video_clips)
        ],
        "inference": inference_summaries,
        "candidate_count": len(all_candidates),
        "selected_optional_count": len(selection.selected),
        "mandatory_count": len(mandatory),
        "timeline": timeline.to_dict(),
        "audio": {
            "music_path": music_directive.path if music_directive else None,
            "mix_with_original": music_directive.mix if music_directive else False,
            "sync_mode": audio_config.sync_mode.value if music_directive else "off",
            "fade_duration": audio_config.fade_duration if music_directive else 0.0,
        },
        "output": {
            "path": str(render_result.output_path.resolve()),
            "duration": render_result.duration,
        },
    }
    result["result_manifest"] = str(result_path.resolve())
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    logger.info("Rendering complete: %s (%.1fs).", render_result.output_path, render_result.duration)
    logger.info("Result manifest written to %s.", result_path.resolve())
    logger.info("AutoCut run completed successfully.")
    return result

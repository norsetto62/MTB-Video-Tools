"""End-to-end production orchestration for AutoCut."""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from ..annotations import load_annotations
from ..data_models import FeatureSequence, FeatureWindows, VideoInfo
from ..dataset.windows import build_windows
from ..editing.audio import render_timeline, render_with_music
from ..ml.inference import infer_windows
from ..motion.features import FeatureExtractor
from ..motion.flow import calculate_optical_flow
from ..selection.budget import MandatoryClip, SelectionCandidate, select_highlight_budget
from ..selection.candidates_generation import CandidateClip, generate_candidates
from ..selection.timeline import Timeline, assemble_timeline
from ..video.probe import get_video_info
from ..video.reader import VideoReader
from ..audio.pipeline import prepare_audio_timeline

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
    settings: object,
) -> FeatureWindows | None:
    """Decode one requested source range and build its temporal feature windows.

    Returns None when the range is too short to produce a complete window.
    Timestamps in the returned windows are relative to the range start; this
    lets candidate generation clamp and merge within that range only.
    """
    feature_rows: list[np.ndarray] = []
    timestamps: list[float] = []
    previous_gray: np.ndarray | None = None
    extractor: FeatureExtractor | None = None

    with VideoReader(
        video_path,
        start=start,
        duration=end - start,
        sample_fps=settings.sample_fps,
    ) as reader:
        for _frame_index, timestamp, frame in reader:
            height, width = frame.shape[:2]
            target_width = min(width, settings.processing_width)
            if target_width != width:
                target_height = max(2, int(round(height * target_width / width)))
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous_gray is not None:
                if extractor is None:
                    extractor = FeatureExtractor(gray.shape[1], gray.shape[0])
                flow = calculate_optical_flow(previous_gray, gray)
                feature_rows.append(extractor.extract_vector(flow))
                timestamps.append(float(timestamp))
            previous_gray = gray

    if len(feature_rows) < 2:
        logger.info(
            "Skipping candidate inference for short range %s [%.3f, %.3f): "
            "not enough sampled frames.",
            video_path, start, end,
        )
        return None

    sequence = FeatureSequence(
        features=np.asarray(feature_rows, dtype=np.float32),
        timestamps=np.asarray(timestamps, dtype=np.float64),
    )
    try:
        windows = build_windows(
            sequence,
            window=settings.common.window_duration,
            stride=settings.common.window_stride,
        )
    except ValueError as exc:
        # A short processing range is not fatal: mandatory clips are still
        # retained by the timeline stage.
        if len(sequence.timestamps) * (1.0 / settings.sample_fps) < settings.common.window_duration:
            logger.info(
                "Skipping candidate inference for short range %s [%.3f, %.3f): %s",
                video_path, start, end, exc,
            )
            return None
        raise
    return windows


def _offset_candidate(candidate: CandidateClip, offset: float) -> CandidateClip:
    """Translate a range-local candidate and its evidence to source timestamps."""
    from ..selection.candidates_generation import WindowEvidence

    return CandidateClip(
        start=candidate.start + offset,
        end=candidate.end + offset,
        duration=candidate.duration,
        tier=candidate.tier,
        mean_score=candidate.mean_score,
        max_score=candidate.max_score,
        window_evidence=tuple(
            WindowEvidence(
                start=window.start + offset,
                end=window.end + offset,
                score=window.score,
                tier=window.tier,
            )
            for window in candidate.window_evidence
        ),
    )


def _config_summary(settings: object) -> dict:
    """Build a stable, JSON-safe subset of the effective run configuration."""
    return {
        "sample_fps": settings.sample_fps,
        "processing_width": settings.processing_width,
        "window_duration": settings.common.window_duration,
        "window_stride": settings.common.window_stride,
        "checkpoint": str(settings.checkpoint),
        "inference_batch_size": settings.inference_batch_size,
        "target_duration": settings.target_duration,
        "min_clip": settings.common.min_clip,
        "max_clip_duration": settings.max_clip_duration,
        "candidate_generation": asdict(settings.candidates),
        "audio": {
            "sync_mode": settings.audio.sync_mode.value,
            "max_clip_duration": settings.audio.max_clip_duration,
            "fade_duration": settings.audio.fade_duration,
        },
    }


def run_pipeline(
    annotation_path: str | Path,
    settings: object,
    *,
    output_path: str | Path | None = None,
) -> dict:
    """Run AutoCut from an annotation manifest through final rendering.

    The annotation intervals define the source ranges to analyze. Mandatory
    intervals are separately protected during selection and are never trimmed
    or synchronized. The returned dictionary is also written as a result JSON.
    """
    annotation = Path(annotation_path)
    if not annotation.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation}")
    if not settings.checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {settings.checkpoint}")

    clips, music_directive = load_annotations(
        annotation,
        min_clip=settings.common.min_clip,
    )
    if not clips:
        raise ValueError(
            f"No processing intervals were found in annotation manifest: {annotation}"
        )

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
        video_metadata[video_name] = info
        for clip in video_clips:
            if clip.start < 0 or clip.end > info.duration + 1e-6:
                raise ValueError(
                    f"Annotation interval [{clip.start:.3f}, {clip.end:.3f}) "
                    f"exceeds video duration {info.duration:.3f}s: {video_name}"
                )

    if music_directive is not None and not Path(music_directive.path).is_file():
        raise FileNotFoundError(f"Background music not found: {music_directive.path}")

    if not clips_by_video:
        raise ValueError("The annotation manifest does not identify any video ranges.")

    all_candidates: list[SelectionCandidate] = []
    inference_summaries: list[dict] = []
    for video_name, video_clips in clips_by_video.items():
        order = video_order[video_name]
        for start, end in _merge_ranges(video_clips):
            windows = _extract_windows(Path(video_name), start, end, settings)
            if windows is None:
                continue
            inference = infer_windows(
                settings.checkpoint,
                windows,
                batch_size=settings.inference_batch_size,
            )
            candidates = generate_candidates(
                inference,
                clip_duration=end - start,
                config=settings.candidates,
            )
            inference_summaries.append({
                "video": video_name,
                "range_start": start,
                "range_end": end,
                "window_count": int(windows.X.shape[0]),
                "candidate_count": len(candidates),
            })
            all_candidates.extend(
                SelectionCandidate(
                    video_name=video_name,
                    video_order=order,
                    candidate=_offset_candidate(candidate, start),
                )
                for candidate in candidates
            )

    mandatory = tuple(
        MandatoryClip(
            video_name=str(Path(clip.video_name).resolve()),
            start=float(clip.start),
            end=float(clip.end),
            video_order=video_order[str(Path(clip.video_name).resolve())],
            sequence_order=index,
        )
        for index, clip in enumerate(clips)
        if clip.mandatory
    )
    selection = select_highlight_budget(
        all_candidates,
        settings.target_duration,
        mandatory,
        min_trimmed_duration=settings.common.min_clip,
        max_clip_duration=settings.max_clip_duration,
    )
    timeline = assemble_timeline(
        selection,
        mandatory,
        max_clip_duration=settings.max_clip_duration,
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
            config=settings.audio,
            cache_dir=settings.cache_dir / "audio",
        )

    destination = (
        Path(output_path)
        if output_path is not None
        else settings.common.output_dir / f"{annotation.stem}_autocut.mp4"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if music_directive is None:
        render_result = render_timeline(timeline, destination)
    else:
        render_result = render_with_music(
            timeline,
            music_directive.path,
            destination,
            config=settings.audio,
            mix_original_audio=music_directive.mix,
        )

    result_path = destination.with_name(f"{destination.stem}_result.json")
    result = {
        "schema_version": 1,
        "status": "completed",
        "annotation_manifest": str(annotation.resolve()),
        "checkpoint": str(settings.checkpoint.resolve()),
        "configuration": _config_summary(settings),
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
            "sync_mode": settings.audio.sync_mode.value if music_directive else "off",
            "fade_duration": settings.audio.fade_duration if music_directive else 0.0,
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
    logger.info("Rendered AutoCut output to %s (%.3fs)", render_result.output_path, render_result.duration)
    return result

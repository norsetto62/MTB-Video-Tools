"""Checkpoint inference on an explicitly selected manifest group."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..annotations import load_training_annotations
from .training import _data_values, _path, _section
from ..data_models import Clip
from ..dataset.manifest import DatasetRecord, load_manifest
from ..dataset.sequence import build_feature_sequence
from ..dataset.windows import build_windows
from ..ml.checkpoint import load_checkpoint, restore_model
from ..ml.scaling import FeatureScaler

logger = logging.getLogger(__name__)


def _predict_record(
    record: DatasetRecord,
    settings: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    scaler: FeatureScaler,
    checkpoint: dict[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    """Extract every complete window from a record and predict its class."""
    annotation_video, _ = load_training_annotations(record.annotation_path)
    if annotation_video.resolve() != record.video_path.resolve():
        raise ValueError(
            f"Manifest entry {record.id!r} points to {record.video_path}, but "
            f"its annotation file declares {annotation_video}."
        )

    from ..video.probe import get_video_info

    info = get_video_info(record.video_path)
    clip = Clip(video_name=record.id, start=0.0, end=info.duration)
    data_config = checkpoint.get("data_config") or {}
    _manifest, configured_fps, configured_width, configured_window, configured_stride = _data_values(settings)
    flow_fps = float(data_config.get("flow_fps", configured_fps))
    processing_width = int(data_config.get("processing_width", configured_width))
    window = float(data_config.get("window", configured_window))
    stride = float(data_config.get("stride", configured_stride))
    paths = _section(settings, "paths")
    cache_dir = _path(paths.get("cache_dir", "data/cache"), name="paths.cache_dir")

    logger.info(
        "Record %s: extracting features at %.3f fps, width %d",
        record.id, flow_fps, processing_width,
    )
    sequence = build_feature_sequence(
        record.video_path,
        clip,
        sample_fps=flow_fps,
        processing_width=processing_width,
        cache_dir=cache_dir,
    )
    windows = build_windows(sequence, window=window, stride=stride)
    X = windows.X

    feature_mask = checkpoint.get("feature_mask")
    if feature_mask is not None:
        mask = np.asarray(feature_mask, dtype=bool)
        if mask.ndim != 1 or mask.size != X.shape[-1]:
            raise ValueError(
                f"Checkpoint feature mask has {mask.size} entries but extracted "
                f"features have {X.shape[-1]} columns."
            )
        X = X[..., mask]

    if X.shape[-1] != int(checkpoint["feature_dim"]):
        raise ValueError(
            f"Checkpoint expects {checkpoint['feature_dim']} features, "
            f"but extraction produced {X.shape[-1]} after applying its mask."
        )

    transformed = scaler.transform(X)
    tensor = torch.from_numpy(np.ascontiguousarray(transformed, dtype=np.float32))
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tensor), batch_size):
            logits = model(tensor[start : start + batch_size])
            probs = torch.softmax(logits, dim=1)
            probabilities.append(probs.cpu().numpy())

    predicted_probabilities = np.concatenate(probabilities)
    if predicted_probabilities.shape[1] == 3:
        training_config = checkpoint.get("training_config") or {}
        threshold = training_config.get("c2_threshold", 0.5)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("Checkpoint c2_threshold must be finite and between 0 and 1.")
        fallback = predicted_probabilities[:, :2].argmax(axis=1)
        predicted_classes = np.where(predicted_probabilities[:, 2] >= float(threshold), 2, fallback)
    else:
        predicted_classes = predicted_probabilities.argmax(axis=1)
    class_mapping = checkpoint.get("class_mapping") or {}
    rows = []
    for index, ((start, end), class_id, probs) in enumerate(
        zip(windows.timestamps, predicted_classes, predicted_probabilities, strict=True)
    ):
        rows.append({
            "window_index": index,
            "start": round(float(start), 6),
            "end": round(float(end), 6),
            "predicted_class": int(class_id),
            "predicted_label": class_mapping.get(str(int(class_id)), str(int(class_id))),
            "probabilities": {
                str(class_index): float(probability)
                for class_index, probability in enumerate(probs)
            },
        })

    return {
        "record_id": record.id,
        "group_id": record.group_id,
        "video": str(record.video_path),
        "duration_seconds": float(info.duration),
        "window_seconds": window,
        "stride_seconds": stride,
        "flow_fps": flow_fps,
        "window_count": len(rows),
        "windows": rows,
    }


def run_validation(
    manifest_path: str | Path,
    settings: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    group_id: str,
    output_path: str | Path | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Run checkpoint-only inference for every record in one manifest group.

    This function never trains or modifies the checkpoint. The scaler and
    temporal window parameters are restored from the checkpoint.
    """
    if not isinstance(group_id, str) or not group_id.strip():
        raise ValueError("group_id must be a non-empty string.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    checkpoint_file = Path(checkpoint_path).expanduser()
    checkpoint = load_checkpoint(checkpoint_file, map_location="cpu")
    scaler_state = checkpoint.get("scaler_state")
    if scaler_state is None:
        scaler = FeatureScaler(enabled=False)
    else:
        scaler = FeatureScaler.from_state_dict(scaler_state)
    if scaler.enabled and (scaler.mean_ is None or scaler.scale_ is None):
        raise ValueError("Checkpoint contains an enabled but unfitted feature scaler.")

    model = restore_model(checkpoint, device="cpu")
    records = load_manifest(manifest_path, verify_files=True)
    data = _section(settings, "data")
    train_groups = data.get("train_groups", ())
    if not isinstance(train_groups, (tuple, list)):
        raise ValueError("data.train_groups must be a list of group IDs.")
    if group_id.strip() in set(train_groups):
        raise ValueError(
            f"Refusing to evaluate training group {group_id.strip()!r}; "
            "choose a group excluded from data.train_groups."
        )
    selected = tuple(record for record in records if record.group_id == group_id.strip())
    if not selected:
        available = sorted({record.group_id for record in records})
        raise ValueError(
            f"No manifest records found for group {group_id!r}. "
            f"Available groups: {', '.join(available)}"
        )

    logger.info(
        "Validating checkpoint %s against group %s (%d records)",
        checkpoint_file, group_id, len(selected),
    )
    record_results = [
        _predict_record(
            record, settings, model=model, scaler=scaler,
            checkpoint=checkpoint, batch_size=batch_size,
        )
        for record in selected
    ]
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_file.resolve()),
        "manifest": str(Path(manifest_path).expanduser().resolve()),
        "group_id": group_id.strip(),
        "records": record_results,
        "total_windows": sum(item["window_count"] for item in record_results),
        "class_mapping": checkpoint.get("class_mapping") or {},
    }

    experiment = _section(settings, "experiment")
    results_dir = _path(
        experiment.get("results_dir", "output/experiments"),
        name="experiment.results_dir",
    )
    destination = (
        Path(output_path).expanduser()
        if output_path is not None
        else results_dir / f"{checkpoint_file.stem}_{group_id.strip()}_predictions.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    result["results_path"] = str(destination)
    logger.info("Saved predictions: %s", destination)
    return result

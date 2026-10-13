"""Checkpoint inference on an explicitly selected manifest group."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .annotations import load_training_annotations
from .config_loader import TrainingRunConfig
from .data_models import Clip
from .dataset.manifest import DatasetRecord, load_manifest
from .dataset.sequence import build_feature_sequence
from .dataset.windows import build_windows
from .ml.checkpoint import load_checkpoint, restore_model
from .ml.scaling import FeatureScaler

logger = logging.getLogger(__name__)


def _predict_record(
    record: DatasetRecord,
    settings: TrainingRunConfig,
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

    from .video.probe import get_video_info

    info = get_video_info(record.video_path)
    clip = Clip(video_name=record.id, start=0.0, end=info.duration)
    data_config = checkpoint.get("data_config") or {}
    flow_fps = float(data_config.get("flow_fps", settings.sample_fps))
    processing_width = int(data_config.get("processing_width", settings.processing_width))
    window = float(data_config.get("window", settings.common.window_duration))
    stride = float(data_config.get("stride", settings.common.window_stride))

    logger.info(
        "Record %s: extracting features at %.3f fps, width %d",
        record.id, flow_fps, processing_width,
    )
    sequence = build_feature_sequence(
        record.video_path,
        clip,
        sample_fps=flow_fps,
        processing_width=processing_width,
        cache_dir=settings.cache_dir,
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
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tensor), batch_size):
            logits = model(tensor[start : start + batch_size])
            probs = torch.softmax(logits, dim=1)
            probabilities.append(probs.cpu().numpy())
            predictions.append(probs.argmax(dim=1).cpu().numpy())

    predicted_classes = np.concatenate(predictions)
    predicted_probabilities = np.concatenate(probabilities)
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
    settings: TrainingRunConfig,
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

    destination = (
        Path(output_path).expanduser()
        if output_path is not None
        else settings.results_dir / f"{checkpoint_file.stem}_{group_id.strip()}_predictions.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\\n") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\\n")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    result["results_path"] = str(destination)
    logger.info("Saved predictions: %s", destination)
    return result

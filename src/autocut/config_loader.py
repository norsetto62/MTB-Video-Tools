"""Load and validate AutoCut run and training YAML configuration.

Paths in YAML are interpreted relative to the current working directory, not
relative to the YAML file. Configuration sections map onto the existing
validated dataclasses wherever possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject
    raise ImportError(
        "YAML configuration requires PyYAML. Install AutoCut with its project "
        "dependencies (for example: pip install -e .)."
    ) from exc

from .audio.config import AudioConfig
from .ml.training import TrainingConfig

@dataclass(frozen=True)
class CommonSettings:
    """Shared settings resolved by the YAML configuration loaders."""

    data_dir: Path
    models_dir: Path
    output_dir: Path
    log_level: str
    min_clip: float
    window_duration: float
    window_stride: float


@dataclass(frozen=True)
class RunConfig:
    """Validated settings needed by the production run pipeline."""

    common: CommonSettings
    cache_dir: Path
    sample_fps: float
    processing_width: int
    video_encoder: str
    checkpoint: Path
    inference_batch_size: int
    max_clip_duration: float
    candidates: Any
    audio: AudioConfig


@dataclass(frozen=True)
class TrainingRunConfig:
    """Validated dataset, split, training and experiment settings."""

    common: CommonSettings
    cache_dir: Path
    manifest: Path
    sample_fps: float
    processing_width: int
    study_name: str
    n_trials: int
    proxy_test_groups: tuple[str, ...]
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    training: TrainingConfig
    experiment_name: str
    checkpoint_dir: Path
    results_dir: Path
    hpo_search_space: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class HpoConfig:
    """Study name, number of trials, proxy test groups  """

    study_name: str
    n_trials: int
    proxy_test_groups: tuple[str, ...]
    c2_threshold: float


def _mapping(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section {where!r} must be a mapping.")
    return dict(value)


def _section(
    root: Mapping[str, Any],
    name: str,
    allowed: set[str],
    *,
    required: bool = False,
) -> dict[str, Any]:
    if name not in root:
        if required:
            raise ValueError(f"Missing required configuration section {name!r}.")
        return {}
    result = _mapping(root[name], where=name)
    unknown = set(result) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown field(s) in {name!r}: {names}.")
    return result


def _keys(root: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = set(root) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown top-level configuration section(s): {names}.")


def _path(value: Any, *, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty path string.")
    return Path(value).expanduser()


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number.")
    result = float(value)
    if result <= 0 or result == float("inf") or result != result:
        raise ValueError(f"{name} must be a finite positive number.")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _groups(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty YAML list of group IDs.")
    groups: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} must contain non-empty string group IDs.")
        group = item.strip()
        if group in groups:
            raise ValueError(f"{name} contains duplicate group ID {group!r}.")
        groups.append(group)
    return tuple(groups)


def _hpo_search_space(value: Any) -> dict[str, dict[str, Any]]:
    """Validate the Optuna distributions supported by the HPO objective."""
    raw = _mapping(value, where="hpo.search_space")
    required = {
        "lr", "weight_decay", "dropout", "class_weight_0",
        "class_weight_2", "hidden_dim", "c2_threshold",
    }
    missing = required - set(raw)
    unknown = set(raw) - required
    if missing or unknown:
        details = []
        if missing:
            details.append("missing: " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown: " + ", ".join(sorted(unknown)))
        raise ValueError("Invalid hpo.search_space parameters (" + "; ".join(details) + ").")

    validated: dict[str, dict[str, Any]] = {}
    for name, value_spec in raw.items():
        spec = _mapping(value_spec, where=f"hpo.search_space.{name}")
        if "choices" in spec:
            if set(spec) != {"choices"}:
                raise ValueError(
                    f"hpo.search_space.{name} with choices may contain only the choices field."
                )
            choices = spec["choices"]
            if not isinstance(choices, list) or len(choices) < 2:
                raise ValueError(f"hpo.search_space.{name}.choices must contain at least two choices.")
            validated[name] = {"choices": choices}
            continue

        unknown_fields = set(spec) - {"low", "high", "log"}
        if unknown_fields or not {"low", "high"}.issubset(spec):
            raise ValueError(
                f"hpo.search_space.{name} must define low/high and optional log, "
                "or define choices."
            )
        low = spec["low"]
        high = spec["high"]
        if (
            isinstance(low, bool) or not isinstance(low, (int, float))
            or isinstance(high, bool) or not isinstance(high, (int, float))
        ):
            raise ValueError(f"hpo.search_space.{name}.low and high must be numbers.")
        low, high = float(low), float(high)
        if not (low < high and low == low and high == high and abs(low) != float("inf") and abs(high) != float("inf")):
            raise ValueError(f"hpo.search_space.{name} requires finite low < high.")
        log = spec.get("log", False)
        if not isinstance(log, bool):
            raise ValueError(f"hpo.search_space.{name}.log must be a boolean.")
        if log and low <= 0:
            raise ValueError(f"hpo.search_space.{name}.low must be positive when log is true.")
        validated[name] = {"low": low, "high": high, "log": log}

    return validated


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = _path(path, name="config path")
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8-sig") as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc
    if raw is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    return _mapping(raw, where="root")


def _video_encoder(value: Any) -> str:
    """Validate the encoder used to re-encode selected timeline clips."""
    if not isinstance(value, str) or value not in {"h264_nvenc", "libx264"}:
        raise ValueError("video.encoder must be h264_nvenc or libx264.")
    return value


def load_run_config(path: str | Path = "config/autocut.yaml") -> RunConfig:
    """Load production settings and map them to existing typed configurations."""
    raw = _load_yaml(path)
    _keys(raw, {"paths", "video", "features", "model", "selection", "audio", "logging"})

    paths = _section(
        raw, "paths",
        {"data_dir", "models_dir", "output_dir", "cache_dir"},
    )
    video = _section(raw, "video", {"sampling_fps", "processing_width", "encoder"})
    features = _section(raw, "features", {"window_duration", "window_stride"})
    model = _section(raw, "model", {"checkpoint", "inference_batch_size"})
    selection = _section(
        raw, "selection",
        {
            "min_clip", "max_clip_duration",
            "class_score_weights", "junk_threshold", "action_threshold",
            "flow_threshold", "score_threshold", "min_candidate_duration",
            "max_gap_to_merge", "padding_seconds", "p2_envelope_decay",
        },
        required=True,
    )
    audio = _section(
        raw, "audio", {"sync_mode", "max_synced_clip_duration", "fade_duration"}
    )
    logging = _section(raw, "logging", {"level"})

    common = CommonSettings(
        data_dir=_path(paths.get("data_dir", "data"), name="paths.data_dir"),
        models_dir=_path(paths.get("models_dir", "models"), name="paths.models_dir"),
        output_dir=_path(paths.get("output_dir", "output"), name="paths.output_dir"),
        log_level=logging.get("level", "INFO"),
        min_clip=selection.get("min_clip", 3.0),
        window_duration=features.get("window_duration", 4.0),
        window_stride=features.get("window_stride", 2.0),
    )
    if not isinstance(common.log_level, str) or common.log_level.upper() not in {
        "CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"
    }:
        raise ValueError("logging.level must be a standard Python logging level.")
    if common.window_duration <= 0 or common.window_stride <= 0:
        raise ValueError("features.window_duration and window_stride must be positive.")
    if common.min_clip <= 0:
        raise ValueError("selection.min_clip must be positive.")

    weights = selection.get("class_score_weights", (0.0, 1.0, 2.5))
    if not isinstance(weights, (list, tuple)) or len(weights) != 3:
        raise ValueError("selection.class_score_weights must contain three numbers.")

    from .selection.candidates_generation import CandidateGenerationConfig

    candidates = CandidateGenerationConfig(
        class_score_weights=tuple(weights),
        junk_threshold=selection.get("junk_threshold", 0.20),
        action_threshold=selection.get("action_threshold", 0.60),
        flow_threshold=selection.get("flow_threshold", 0.40),
        # min_clip is the sole final candidate-duration filter. The old
        # min_candidate_duration key remains accepted for config compatibility
        # but no longer controls filtering independently.
        min_candidate_duration=common.min_clip,
        max_gap_to_merge=selection.get("max_gap_to_merge", 3.0),
        padding_seconds=selection.get("padding_seconds", 2.0),
        p2_envelope_decay=selection.get("p2_envelope_decay", 0.0),
    )
    audio_config = AudioConfig(
        sync_mode=audio.get("sync_mode", "off"),
        max_synced_clip_duration=audio.get("max_synced_clip_duration", 0.0),
        fade_duration=audio.get("fade_duration", 2.0),
    )

    return RunConfig(
        common=common,
        cache_dir=_path(paths.get("cache_dir", "data/cache"), name="paths.cache_dir"),
        sample_fps=_positive_float(video.get("sampling_fps", 2.0), name="video.sampling_fps"),
        processing_width=_positive_int(
            video.get("processing_width", 480), name="video.processing_width"
        ),
        video_encoder=_video_encoder(video.get("encoder", "h264_nvenc")),
        checkpoint=_path(model.get("checkpoint", "models/best_model.pt"), name="model.checkpoint"),
        inference_batch_size=_positive_int(
            model.get("inference_batch_size", 256), name="model.inference_batch_size"
        ),
        max_clip_duration=_nonnegative_float(
            selection.get("max_clip_duration", 0.0), name="selection.max_clip_duration"
        ),
        candidates=candidates,
        audio=audio_config,
    )
  

def load_training_config(path: str | Path = "config/train.yaml") -> TrainingRunConfig:
    """Load training YAML and construct the existing TrainingConfig dataclass."""
    raw = _load_yaml(path)
    _keys(raw, {"paths", "hpo", "data", "training", "experiment", "logging"})

    paths = _section(raw, "paths", {"data_dir", "models_dir", "cache_dir"})
    hpo = _section(
        raw, "hpo",
        {
            "study_name", "n_trials", "proxy_test_groups", "search_space",
        },
        required=False,
    )
    data = _section(
        raw, "data",
        {
            "manifest", "sampling_fps", "processing_width", "window_duration",
            "window_stride", "train_groups", "validation_groups",
        },
        required=True,
    )
    training_values = _section(
        raw, "training",
        {
            "epochs", "lr", "batch_size", "hidden_dim", "classifier_mid_dim",
            "dropout", "num_classes", "use_scaling", "patience", "min_delta",
            "weight_decay", "seed", "class_weights",
        },
    )
    experiment = _section(
        raw, "experiment", {"name", "checkpoint_dir", "results_dir"}
    )
    logging = _section(raw, "logging", {"level"})

    level = logging.get("level", "INFO")
    if not isinstance(level, str) or level.upper() not in {
        "CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"
    }:
        raise ValueError("logging.level must be a standard Python logging level.")

    common = CommonSettings(
        data_dir=_path(paths.get("data_dir", "data"), name="paths.data_dir"),
        models_dir=_path(paths.get("models_dir", "models"), name="paths.models_dir"),
        output_dir=Path("output"),
        log_level=level.upper(),
        min_clip=3.0,
        window_duration=_positive_float(
            data.get("window_duration", 4.0), name="data.window_duration"
        ),
        window_stride=_positive_float(
            data.get("window_stride", 2.0), name="data.window_stride"
        ),
    )

    if hpo:
        study_name = hpo.get("study_name")
        if not isinstance(study_name, str) or not study_name.strip():
            raise ValueError("hpo.study_name must be a non-empty string.")
        n_trials = _positive_int(hpo.get("n_trials"), name="hpo.n_trials")
        proxy_test_groups = _groups(hpo.get("proxy_test_groups"), name="hpo.proxy_test_groups")
        if "search_space" not in hpo:
            raise ValueError("Missing required configuration field 'hpo.search_space'.")
        hpo_search_space = _hpo_search_space(hpo["search_space"])
    else:
        study_name = ''
        n_trials = 0
        proxy_test_groups = tuple()
        hpo_search_space = {}

    train_groups = _groups(data.get("train_groups"), name="data.train_groups")
    validation_groups = _groups(
        data.get("validation_groups"), name="data.validation_groups"
    )
    overlap = set(train_groups) & set(validation_groups)
    if overlap:
        raise ValueError(
            "Training and validation groups must be disjoint; overlap: "
            + ", ".join(sorted(overlap))
        )

    training = TrainingConfig(**training_values)
    name = experiment.get("name", "baseline")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("experiment.name must be a non-empty string.")

    return TrainingRunConfig(
        common=common,
        cache_dir=_path(paths.get("cache_dir", "data/cache"), name="paths.cache_dir"),
        manifest=_path(
            data.get("manifest", "data/manifest/annotations_manifest.yaml"),
            name="data.manifest",
        ),
        sample_fps=_positive_float(
            data.get("sampling_fps", 2.0), name="data.sampling_fps"
        ),
        processing_width=_positive_int(
            data.get("processing_width", 480), name="data.processing_width"
        ),
        study_name=study_name,
        n_trials=n_trials,
        proxy_test_groups=proxy_test_groups,
        train_groups=train_groups,
        validation_groups=validation_groups,
        training=training,
        experiment_name=name.strip(),
        checkpoint_dir=_path(
            experiment.get("checkpoint_dir", "models/experiments"),
            name="experiment.checkpoint_dir",
        ),
        results_dir=_path(
            experiment.get("results_dir", "output/experiments"),
            name="experiment.results_dir",
        ),
        hpo_search_space=hpo_search_space,
    )


def _nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative number.")
    result = float(value)
    if result < 0 or result == float("inf") or result != result:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return result

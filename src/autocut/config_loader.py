"""Generic immutable configuration loading.

This module deliberately knows nothing about AutoCut domain classes. It turns
YAML mappings into immutable values and provides copy-on-write overrides.
Application-specific validation and object construction belong to consumers.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject
    raise ImportError(
        "YAML configuration requires PyYAML. Install project dependencies "
        "(for example: pip install -e .)."
    ) from exc


class _DeleteSentinel:
    __slots__ = ()

    def __repr__(self) -> str:
        return "DELETE"


DELETE = _DeleteSentinel()


def _freeze(value: Any) -> Any:
    """Recursively freeze containers without interpreting domain semantics."""
    if isinstance(value, Config):
        return value
    if isinstance(value, Mapping):
        return Config(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Return ordinary containers, suitable for serialization or edits."""
    if isinstance(value, Config):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return {_thaw(item) for item in value}
    return value


class Config(Mapping[str, Any]):
    """A recursively immutable mapping with convenient attribute access.

    Values are generic Python scalars, nested Config instances, and tuples.
    Use with_overrides() to create a modified copy.
    """

    __slots__ = ("_data",)

    def __init__(self, values: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        merged = dict(values or {})
        merged.update(kwargs)
        if not all(isinstance(key, str) for key in merged):
            raise TypeError("Configuration mapping keys must be strings.")
        object.__setattr__(
            self, "_data",
            MappingProxyType({key: _freeze(value) for key, value in merged.items()}),
        )

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Config is immutable; use with_overrides().")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Config is immutable; use with_overrides().")

    def to_dict(self) -> dict[str, Any]:
        """Return a mutable deep copy of the configuration values."""
        return _thaw(self)

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Config":
        """Return a copy with dotted-path overrides applied.

        Keys are dotted paths such as "video.fps". DELETE removes a key.
        The original object is never modified.
        """
        result = self.to_dict()
        for dotted_path, value in overrides.items():
            if not isinstance(dotted_path, str) or not dotted_path.strip():
                raise ValueError("Override paths must be non-empty strings.")
            parts = dotted_path.split(".")
            if any(not part for part in parts):
                raise ValueError(f"Invalid dotted override path: {dotted_path!r}.")
            target = result
            skip = False
            for part in parts[:-1]:
                if part not in target:
                    if value is DELETE:
                        skip = True
                        break
                    target[part] = {}
                if not isinstance(target[part], dict):
                    raise ValueError(
                        f"Cannot apply override {dotted_path!r}: "
                        f"{part!r} is not a configuration section."
                    )
                target = target[part]
            if skip:
                continue
            leaf = parts[-1]
            if value is DELETE:
                target.pop(leaf, None)
            else:
                target[leaf] = _thaw(_freeze(value))
        return Config(result)

    def override(self, dotted_path: str, value: Any) -> "Config":
        """Return a copy with one dotted-path override applied."""
        return self.with_overrides({dotted_path: value})


def load_config(path: str | Path) -> Config:
    """Load YAML as generic immutable values, without app-specific validation."""
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("Configuration path must be a non-empty path string.")
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8-sig") as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc
    if raw is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    if not isinstance(raw, Mapping):
        raise ValueError("Configuration root must be a mapping.")
    return Config(raw)

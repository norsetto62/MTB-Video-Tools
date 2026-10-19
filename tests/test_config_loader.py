from pathlib import Path

import pytest

from autocut.config_loader import Config, DELETE, load_config


def test_config_supports_mapping_and_attribute_access():
    config = Config({"video": {"fps": 2, "resolutions": [480, 720]}, "enabled": True})

    assert config.video.fps == 2
    assert config["video"]["fps"] == 2
    assert config.video.resolutions == (480, 720)
    assert config.enabled is True


def test_config_is_recursively_immutable():
    config = Config({"nested": {"value": 1}, "list_values": [1, 2]})

    with pytest.raises(AttributeError, match="immutable"):
        config.nested.value = 2
    with pytest.raises(AttributeError, match="immutable"):
        config.new_value = 3
    with pytest.raises(TypeError):
        config.nested["value"] = 2
    with pytest.raises(AttributeError):
        config.list_values.append(3)


def test_overrides_are_copy_on_write_and_support_delete():
    original = Config({"video": {"fps": 2, "encoder": "h264"}, "logging": {"level": "INFO"}})

    changed = original.with_overrides({
        "video.fps": 3,
        "video.encoder": DELETE,
        "selection.min_clip": 4,
    })

    assert original.video.fps == 2
    assert original.video.encoder == "h264"
    assert "selection" not in original
    assert changed.video.fps == 3
    assert "encoder" not in changed.video
    assert changed.selection.min_clip == 4


def test_override_rejects_path_through_scalar():
    config = Config({"video": {"fps": 2}})

    with pytest.raises(ValueError, match="not a configuration section"):
        config.override("video.fps.value", 3)


def test_load_config_loads_generic_values_without_domain_conversion(tmp_path: Path):
    path = tmp_path / "settings.yaml"
    path.write_text("audio:\n  sync_mode: beat\n  channels: [left, right]\n", encoding="utf-8")

    config = load_config(path)

    assert isinstance(config, Config)
    assert config.audio.sync_mode == "beat"
    assert config.audio.channels == ("left", "right")
    assert config.to_dict() == {"audio": {"sync_mode": "beat", "channels": ["left", "right"]}}


def test_config_does_not_alias_mutable_input_values():
    nested = {"items": [1, {"value": 2}]}
    config = Config(nested)

    nested["items"][1]["value"] = 99
    nested["items"].append(3)

    assert config.items == (1, Config({"value": 2}))
    assert config.to_dict() == {"items": [1, {"value": 2}]}


def test_override_freezes_inserted_values_without_mutating_caller_data():
    config = Config({"section": {"existing": True}})
    inserted = {"values": [1, 2]}

    changed = config.override("section.added", inserted)
    inserted["values"].append(3)

    assert config.section == Config({"existing": True})
    assert changed.section.added.values == (1, 2)

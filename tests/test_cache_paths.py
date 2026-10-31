from pathlib import Path

import pytest

from autocut.config_loader import Config
from autocut.dataset.cache_paths import audio_cache_dir, cache_root, feature_cache_dir


def test_default_cache_directories():
    settings = Config({"paths": {}})

    assert cache_root(settings) == Path("data/cache")
    assert feature_cache_dir(settings) == Path("data/cache/features")
    assert audio_cache_dir(settings) == Path("data/cache/audio")


def test_custom_cache_root_is_shared_by_feature_and_audio_caches(tmp_path):
    settings = {"paths": {"cache_dir": str(tmp_path / "custom-cache")}}

    assert cache_root(settings) == tmp_path / "custom-cache"
    assert feature_cache_dir(settings) == tmp_path / "custom-cache" / "features"
    assert audio_cache_dir(settings) == tmp_path / "custom-cache" / "audio"


@pytest.mark.parametrize("value", ["", None, 12])
def test_invalid_cache_root_is_rejected(value):
    with pytest.raises(ValueError, match="paths.cache_dir"):
        feature_cache_dir({"paths": {"cache_dir": value}})

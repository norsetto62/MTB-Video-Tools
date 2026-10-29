"""Tests for LightGBM ranking HPO objective selection."""
import pytest

from autocut.config_loader import Config
from autocut.ml.model import LightGBMConfig
from autocut.pipeline.lightgbm_hpo import _suggest, _validated_search_space


def test_lightgbm_config_accepts_both_ranking_objectives():
    assert LightGBMConfig(objective="lambdarank").objective == "lambdarank"
    assert LightGBMConfig(objective="rank_xendcg").objective == "rank_xendcg"


def test_lightgbm_config_rejects_non_ranking_objective():
    with pytest.raises(ValueError, match="rank_xendcg"):
        LightGBMConfig(objective="multiclass")


def test_hpo_search_space_includes_both_objectives_by_default():
    space = _validated_search_space(Config({}))
    assert space["objective"]["choices"] == ["lambdarank", "rank_xendcg"]


def test_hpo_search_space_requires_both_objective_choices():
    settings = Config({
        "lightgbm_hpo": {
            "search_space": {
                "objective": {"choices": ["lambdarank"]},
            }
        }
    })
    with pytest.raises(ValueError, match="exactly these choices"):
        _validated_search_space(settings)


def test_suggest_uses_categorical_objective():
    class Trial:
        def __init__(self):
            self.calls = []

        def suggest_categorical(self, name, choices):
            self.calls.append((name, choices))
            return "rank_xendcg"

    trial = Trial()
    params = _suggest(trial, {"objective": {"choices": ["lambdarank", "rank_xendcg"]}})
    assert params == {"objective": "rank_xendcg"}
    assert trial.calls == [("objective", ["lambdarank", "rank_xendcg"])]

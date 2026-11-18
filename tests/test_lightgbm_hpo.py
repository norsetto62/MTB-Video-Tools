"""Tests for LightGBM ranking HPO objective selection."""
import pytest

from autocut.config_loader import Config
from autocut.ml.lightgbm_model import LightGBMConfig
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
    assert "learning_rate" in space
    assert "num_leaves" in space


def test_hpo_search_space_rejects_invalid_objective_choices():
    settings = Config({
        "hpo": {
            "search_space": {
                "objective": {"choices": ["lambdarank"]},
            }
        }
    })
    with pytest.raises(ValueError, match="objective choices"):
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


def test_hpo_custom_search_space_overrides_without_dropping_defaults():
    settings = Config({
        "hpo": {
            "search_space": {
                "learning_rate": {"low": 0.02, "high": 0.04},
            }
        }
    })
    space = _validated_search_space(settings)
    assert space["learning_rate"] == {"low": 0.02, "high": 0.04, "log": False}
    assert space["objective"]["choices"] == ["lambdarank", "rank_xendcg"]
    assert "num_leaves" in space



def test_hpo_search_space_accepts_scalar_fixed_values():
    settings = Config({
        "hpo": {
            "search_space": {
                "bagging_frequency": 1,
                "lambda_l1": 0.25,
            }
        }
    })
    space = _validated_search_space(settings)
    assert space["bagging_frequency"] == {"fixed": 1}
    assert space["lambda_l1"] == {"fixed": 0.25}



def test_hpo_section_custom_ranges_are_used_verbatim():
    settings = Config({
        "hpo": {
            "search_space": {
                "max_depth": {"low": 4, "high": 12, "step": 1},
                "num_leaves": {"low": 20, "high": 160},
                "min_data_in_leaf": {"low": 50, "high": 200},
                "bagging_frequency": {"choices": [1, 3, 5]},
                "lambda_l1": {"low": 0.01, "high": 1.0, "log": True},
                "lambda_l2": {"low": 0.001, "high": 10.0, "log": True},
            }
        }
    })
    space = _validated_search_space(settings)
    assert space["max_depth"] == {"low": 4, "high": 12, "log": False, "step": 1}
    assert space["num_leaves"] == {"low": 20, "high": 160, "log": False}
    assert space["min_data_in_leaf"] == {"low": 50, "high": 200, "log": False}
    assert space["bagging_frequency"] == {"choices": [1, 3, 5]}
    assert space["lambda_l1"] == {"low": 0.01, "high": 1.0, "log": True}
    assert space["lambda_l2"] == {"low": 0.001, "high": 10.0, "log": True}


def test_hpo_rejects_legacy_config_section():
    settings = Config({
        "lightgbm_hpo": {
            "search_space": {"num_leaves": {"low": 20, "high": 160}}
        }
    })
    with pytest.raises(ValueError, match="obsolete"):
        _validated_search_space(settings)


def test_hpo_search_space_rejects_unknown_options():
    settings = Config({
        "hpo": {
            "search_space": {
                "num_leaves": {"low": 20, "high": 160, "unknown_option": True}
            }
        }
    })
    with pytest.raises(ValueError, match="Unknown search-space option"):
        _validated_search_space(settings)


def test_hpo_search_space_accepts_fixed_objective():
    settings = Config({
        "hpo": {"search_space": {"objective": {"fixed": "rank_xendcg"}}}
    })
    space = _validated_search_space(settings)
    assert space["objective"] == {"fixed": "rank_xendcg"}

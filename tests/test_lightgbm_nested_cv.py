"""Tests for nested evaluation and final parameter selection in LightGBM HPO."""
from types import SimpleNamespace

import pytest

optuna = pytest.importorskip("optuna")

from autocut.pipeline import hpo


def test_hpo_keeps_outer_group_out_of_inner_hpo_and_reports_video_ids(
    monkeypatch, tmp_path
):
    prepared = {f"video_{name}": object() for name in ("a", "b", "c")}
    groups = {
        "video_a": "group_a",
        "video_b": "group_b",
        "video_c": "group_c",
    }
    monkeypatch.setattr(
        hpo,
        "prepare_lightgbm_manifest_data",
        lambda *args, **kwargs: (
            prepared,
            groups,
            {
                "prepared_once": True,
                "feature_normalization": "per_video_median_iqr_all_windows",
            },
        ),
    )

    inner_group_sets = []
    lovo_call_count = 0

    def fake_lovo(video_data, group_by_video, *, config, collect_evaluation_history=True, data_config=None):
        nonlocal lovo_call_count
        lovo_call_count += 1
        inner_group_sets.append(set(group_by_video.values()))
        return tuple(
            SimpleNamespace(
                held_out_group=group,
                summary={
                    "validation_metrics": {
                        "events": 2,
                        "hit_events": 1,
                        "per_video": {"0": {"recall": 0.5}}
                    }
                },
            )
            for group in sorted(set(group_by_video.values()))
        )

    def fake_outer_fold(
        video_data, group_by_video, *, held_out_group, config, collect_evaluation_history=True, data_config=None
    ):
        validation_ids = tuple(sorted(
            video_id for video_id, group in group_by_video.items()
            if group == held_out_group
        ))
        training_ids = tuple(sorted(set(video_data) - set(validation_ids)))
        result = SimpleNamespace(
            held_out_group=held_out_group,
            train_video_ids=training_ids,
            validation_video_ids=validation_ids,
            summary={
                "validation_metrics": {
                    "segment_recall_at_5": 0.75,
                    "events": 4,
                    "hit_events": 3,
                    "per_video": {
                        str(index): {"recall": 0.75}
                        for index, _ in enumerate(validation_ids)
                    },
                },
                "evaluation_history": {},
            },
        )
        return object(), result

    monkeypatch.setattr(hpo, "train_lightgbm_lovo", fake_lovo)
    monkeypatch.setattr(hpo, "train_lightgbm_group_fold", fake_outer_fold)

    settings = {
        "training": {},
        "hpo": {
            "n_trials": 1,
            "progress": False,
            "output_dir": str(tmp_path),
            "storage_path": str(tmp_path / "hpo.sqlite3"),
            "study_name": "test_nested",
            "search_space": {
                "objective": {"fixed": "lambdarank"},
                "num_iterations": {"fixed": 7},
            },
        },
    }
    result = hpo.run_hpo("manifest.yaml", settings)

    assert len(inner_group_sets) == 4
    assert lovo_call_count == 4
    assert inner_group_sets[-1] == {"group_a", "group_b", "group_c"}
    assert result["best_params"]["num_iterations"] == 7
    assert result["final_parameter_selection"]["purpose"] == "production_parameter_selection_only"
    assert result["study_name"] == "test_nested"
    assert (tmp_path / "hpo.sqlite3").is_file()
    persisted = optuna.load_study(
        study_name=result["study_name"],
        storage=f"sqlite:///{(tmp_path / 'hpo.sqlite3').as_posix()}",
    )
    assert len(persisted.trials) == 1
    assert persisted.best_trial.user_attrs["metric"] == "nms_pooled_segment_recall_at_5"
    assert inner_group_sets == [
        {"group_b", "group_c"},
        {"group_a", "group_c"},
        {"group_a", "group_b"},
        {"group_a", "group_b", "group_c"},
    ]
    assert result["outer_fold_count"] == 3
    assert result["mean_outer_segment_recall_at_5"] == pytest.approx(0.75)
    assert result["mean_outer_group_segment_recall_at_5"] == pytest.approx(0.75)
    assert result["evaluated_outer_video_count"] == 3
    for fold in result["outer_folds"]:
        assert fold["outer_per_video_segment_recall_at_5"] == {
            f"video_{fold['outer_group'][-1]}": 0.75
        }
        assert fold["inner_best_params"]["num_iterations"] == 7
        assert fold["outer_group"] not in fold["inner_groups"]
    assert (tmp_path / "lightgbm_hpo_nested_segment_recall_at_5_median_iqr_all_windows.json").is_file()

    # A second invocation with the same study identity appends trials.
    hpo.run_hpo("manifest.yaml", settings)
    resumed = optuna.load_study(
        study_name=result["study_name"],
        storage=f"sqlite:///{(tmp_path / 'hpo.sqlite3').as_posix()}",
    )
    assert len(resumed.trials) == 2

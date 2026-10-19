import pytest

from autocut.pipeline.lovo import _resolve_group_splits


def test_proxy_splits_use_only_requested_test_and_training_groups():
    splits = _resolve_group_splits(
        ("proxy_a", "proxy_b", "train_a", "train_b", "unused"),
        test_groups=("proxy_a", "proxy_b"),
        train_groups=("train_a", "train_b"),
    )

    assert splits == (
        ("proxy_a", ("train_a", "train_b")),
        ("proxy_b", ("train_a", "train_b")),
    )
    for test_group, training_groups in splits:
        assert test_group not in training_groups
        assert "unused" not in training_groups


def test_proxy_splits_reject_overlap_between_test_and_training_groups():
    with pytest.raises(ValueError, match="excluded from training"):
        _resolve_group_splits(
            ("proxy_a", "train_a", "train_b"),
            test_groups=("proxy_a",),
            train_groups=("proxy_a", "train_a"),
        )


def test_proxy_splits_require_two_training_groups_for_inner_epoch_selection():
    with pytest.raises(ValueError, match="at least two training groups"):
        _resolve_group_splits(
            ("proxy_a", "train_a"),
            test_groups=("proxy_a",),
            train_groups=("train_a",),
        )

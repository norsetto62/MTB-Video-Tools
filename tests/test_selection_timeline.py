"""Tests for Phase 10 timeline assembly."""
import logging

from autocut.selection.budget import MandatoryClip, select_highlight_budget
from autocut.selection.timeline import assemble_timeline
from autocut.selection.candidates_generation import CandidateClip
from autocut.selection.budget import SelectionCandidate


def candidate(video, order, start, end, score=1.0):
    return SelectionCandidate(
        video_name=video,
        video_order=order,
        candidate=CandidateClip(
            start=start, end=end, duration=end-start, tier="moderate_interest",
            mean_score=score, max_score=score,
        ),
    )


def test_timeline_restores_source_order_and_chronology():
    later = candidate("video1.mp4", 0, 20, 24, 2.0)
    earlier = candidate("video1.mp4", 0, 3, 7, 1.0)
    second_video = candidate("video2.mp4", 1, 3, 7, 1.5)
    mandatory = [MandatoryClip("video2.mp4", 0, 2, video_order=1, sequence_order=1)]
    selection = select_highlight_budget([later, earlier, second_video], 30, mandatory)
    timeline = assemble_timeline(selection, mandatory)
    assert [(c.video_name, c.source_start, c.mandatory) for c in timeline.clips] == [
        ("video1.mp4", 3, False),
        ("video1.mp4", 20, False),
        ("video2.mp4", 0, True),
        ("video2.mp4", 3, False),
    ]
    assert timeline.duration == sum(c.duration for c in timeline.clips)


def test_timeline_warns_if_actual_duration_exceeds_target(caplog):
    mandatory = [MandatoryClip("v1.mp4", 0, 12, video_order=0, sequence_order=0)]
    selection = select_highlight_budget([], 10, mandatory)
    with caplog.at_level(logging.WARNING):
        timeline = assemble_timeline(selection, mandatory)
    assert timeline.duration == 12
    assert timeline.over_budget_by == 2
    assert "exceeds target" in caplog.text


def test_timeline_serialization_is_json_friendly():
    mandatory = [MandatoryClip("v1.mp4", 0, 2, video_order=0, sequence_order=0)]
    selection = select_highlight_budget([], 5, mandatory)
    data = assemble_timeline(selection, mandatory).to_dict()
    assert data["schema_version"] == 1
    assert data["clips"][0]["mandatory"] is True
    assert data["duration"] == 2



def test_timeline_max_clip_validation_applies_to_optional_clips_only():
    import pytest

    mandatory = [MandatoryClip("must.mp4", 0, 10, video_order=0)]
    selection = select_highlight_budget(
        [candidate("optional.mp4", 1, 0, 4)], 20, mandatory
    )
    with pytest.raises(ValueError, match="exceeds max_clip_duration"):
        assemble_timeline(selection, mandatory, max_clip_duration=3)


def test_timeline_max_clip_duration_does_not_reject_long_mandatory_clips():
    mandatory = [MandatoryClip("must.mp4", 0, 10, video_order=0)]
    selection = select_highlight_budget([], 20, mandatory)
    timeline = assemble_timeline(selection, mandatory, max_clip_duration=3)
    assert len(timeline.clips) == 1
    assert timeline.clips[0].duration == 10
    assert timeline.clips[0].mandatory is True


def test_timeline_rejects_invalid_max_clip_duration():
    import pytest

    with pytest.raises(ValueError, match="max_clip_duration"):
        assemble_timeline(select_highlight_budget([], 10), max_clip_duration=-1)

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
    second_video = candidate("video2.mp4", 1, 1, 5, 1.5)
    mandatory = [MandatoryClip("video2.mp4", 0, 2, sequence_order=1)]
    selection = select_highlight_budget([later, earlier, second_video], 30, mandatory)
    timeline = assemble_timeline(selection, mandatory)
    assert [(c.video_name, c.source_start, c.mandatory) for c in timeline.clips] == [
        ("video1.mp4", 3, False),
        ("video1.mp4", 20, False),
        ("video2.mp4", 0, True),
        ("video2.mp4", 1, False),
    ]
    assert timeline.duration == sum(c.duration for c in timeline.clips)


def test_timeline_warns_if_actual_duration_exceeds_target(caplog):
    mandatory = [MandatoryClip("v1.mp4", 0, 12, sequence_order=0)]
    selection = select_highlight_budget([], 10, mandatory)
    with caplog.at_level(logging.WARNING):
        timeline = assemble_timeline(selection, mandatory)
    assert timeline.duration == 12
    assert timeline.over_budget_by == 2
    assert "exceeds target" in caplog.text


def test_timeline_serialization_is_json_friendly():
    mandatory = [MandatoryClip("v1.mp4", 0, 2, sequence_order=0)]
    selection = select_highlight_budget([], 5, mandatory)
    data = assemble_timeline(selection, mandatory).to_dict()
    assert data["schema_version"] == 1
    assert data["clips"][0]["mandatory"] is True
    assert data["duration"] == 2

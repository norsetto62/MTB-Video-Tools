"""Tests for budget-free timeline assembly."""
import json
from autocut.selection.selection import MandatoryClip, SelectionCandidate, SelectionResult
from autocut.selection.timeline import assemble_timeline
from autocut.selection.candidates_generation import CandidateClip

def candidate(video, order, start, end, score=1.0, annotation_order=None):
    return SelectionCandidate(video, order, CandidateClip(start, end, end-start, score, score), annotation_order)

def test_timeline_restores_source_order_and_chronology():
    later = candidate("video1.mp4", 0, 20, 24, 2.0)
    earlier = candidate("video1.mp4", 0, 3, 7, 1.0)
    second_video = candidate("video2.mp4", 1, 3, 7, 1.5)
    mandatory = [MandatoryClip("video2.mp4", 0, 2, video_order=1, sequence_order=1)]
    selection = SelectionResult((later, earlier, second_video))
    timeline = assemble_timeline(selection, mandatory)
    assert [(c.video_name, c.source_start, c.mandatory) for c in timeline.clips] == [
        ("video1.mp4", 3, False), ("video1.mp4", 20, False),
        ("video2.mp4", 0, True), ("video2.mp4", 3, False)]
    assert timeline.duration == sum(c.duration for c in timeline.clips)

def test_timeline_duration_is_derived_and_has_no_budget_metadata():
    mandatory = [MandatoryClip("v1.mp4", 0, 12, video_order=0, sequence_order=0)]
    timeline = assemble_timeline(SelectionResult(()), mandatory)
    data = timeline.to_dict()
    assert timeline.duration == 12
    assert "target_duration" not in data
    assert "over_budget_by" not in data
    assert data["duration"] == 12
    assert json.dumps(data)

def test_timeline_serialization_preserves_provenance():
    item = candidate("v1.mp4", 0, 2, 6, annotation_order=7)
    data = assemble_timeline(SelectionResult((item,))).to_dict()
    assert data["schema_version"] == 1
    assert data["clips"][0]["mandatory"] is False
    assert data["clips"][0]["annotation_order"] == 7

def test_timeline_interleaves_mandatory_and_optional_by_manifest_order():
    first = candidate("video2.mp4", 1, 10, 14, annotation_order=0)
    mandatory = [MandatoryClip("video1.mp4", 2, 6, video_order=0, sequence_order=1)]
    third = candidate("video1.mp4", 0, 7, 11, annotation_order=2)
    timeline = assemble_timeline(SelectionResult((first, third)), mandatory)
    assert [(c.video_name, c.source_start, c.annotation_order, c.mandatory) for c in timeline.clips] == [
        ("video2.mp4", 10, 0, False), ("video1.mp4", 2, 1, True), ("video1.mp4", 7, 2, False)]

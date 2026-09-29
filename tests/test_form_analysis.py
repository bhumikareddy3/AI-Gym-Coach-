import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.exercise_rules import FormResult, FormIssue
from app.pose_confidence import compute_pose_confidence
from app.temporal_analyzer import TemporalMetrics
from app.form_analysis import build_form_analysis
from tests.helpers import make_landmark_frame


def test_build_form_analysis_schema_matches_spec():
    lf = make_landmark_frame(default_visibility=0.9)
    pose_conf = compute_pose_confidence(lf, exercise="squat")
    form_result = FormResult(score=82.0, issues=[
        FormIssue(code="knee_alignment", message="Knees caving in", severity="medium"),
    ])
    temporal = TemporalMetrics(
        driving_angle_key="knee", angular_velocity_deg_s=40.0,
        movement_quality="controlled", tempo_label="normal", samples_in_window=10,
    )

    analysis = build_form_analysis(
        exercise="squat", form_result=form_result, pose_conf=pose_conf,
        rep_count=8, range_of_motion_pct=87.0, avg_tempo_sec=2.4, temporal=temporal,
    )
    d = analysis.to_dict()

    assert d["exercise"] == "squat"
    assert d["form_score"] == 82.0
    assert d["rep_count"] == 8
    assert d["range_of_motion"] == 87.0
    assert d["tempo"] == 2.4
    assert d["movement_quality"] == "controlled"
    assert len(d["issues"]) == 1
    assert d["issues"][0]["type"] == "knee_alignment"
    assert 0.0 <= d["issues"][0]["confidence"] <= 1.0
    # issue confidence should track pose confidence, not be hardcoded
    assert d["issues"][0]["confidence"] == d["pose_confidence"]


def test_build_form_analysis_without_temporal():
    lf = make_landmark_frame(default_visibility=0.9)
    pose_conf = compute_pose_confidence(lf, exercise="plank")
    form_result = FormResult(score=100.0, issues=[])
    analysis = build_form_analysis(
        exercise="plank", form_result=form_result, pose_conf=pose_conf,
        rep_count=0, range_of_motion_pct=0.0, avg_tempo_sec=0.0, temporal=None,
    )
    assert analysis.movement_quality == "unknown"
    assert analysis.issues == []

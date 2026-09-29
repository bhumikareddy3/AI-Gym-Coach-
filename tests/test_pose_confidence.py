import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pose_confidence import compute_pose_confidence
from tests.helpers import make_landmark_frame


def test_high_confidence_full_body_visible():
    lf = make_landmark_frame(default_visibility=0.95)
    conf = compute_pose_confidence(lf, exercise="squat")
    assert conf.level == "high"
    assert conf.reliable is True
    assert conf.message is None


def test_low_confidence_when_lower_body_occluded():
    lf = make_landmark_frame(visibility={
        "LEFT_HIP": 0.1, "RIGHT_HIP": 0.1,
        "LEFT_KNEE": 0.1, "RIGHT_KNEE": 0.1,
        "LEFT_ANKLE": 0.1, "RIGHT_ANKLE": 0.1,
    })
    conf = compute_pose_confidence(lf, exercise="squat")
    assert conf.level == "low"
    assert conf.reliable is False
    assert conf.message is not None
    assert "lower body" in conf.reason.lower() or "full body" in conf.reason.lower()


def test_medium_confidence_band():
    # push visibility just into the medium band (0.5 <= score < 0.7)
    lf = make_landmark_frame(default_visibility=0.6)
    conf = compute_pose_confidence(lf, exercise="squat")
    assert conf.level == "medium"
    assert conf.reliable is True  # medium is still usable, just not "high"


def test_exercise_specific_landmarks_affect_score():
    # wrists/elbows are only weighted for pushup/bicep_curl, not squat
    lf = make_landmark_frame(visibility={"LEFT_WRIST": 0.0, "RIGHT_WRIST": 0.0})
    squat_conf = compute_pose_confidence(lf, exercise="squat")
    curl_conf = compute_pose_confidence(lf, exercise="bicep_curl")
    assert squat_conf.score > curl_conf.score


def test_reason_flags_out_of_frame():
    # whole body pushed to the frame edge and correspondingly low-visibility
    # -> should read as "out of frame", not a generic occlusion message
    edge_overrides = {name: (0.01, 0.99) for name in (
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
    )}
    lf = make_landmark_frame(overrides=edge_overrides, default_visibility=0.1)
    conf = compute_pose_confidence(lf, exercise="squat")
    assert conf.reliable is False
    assert "frame" in conf.reason.lower()

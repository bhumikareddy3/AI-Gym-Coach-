"""
feature_engineering.py
-----------------------
Turns raw MediaPipe landmarks into the two things every downstream module
needs:

  1. `compute_joint_angles`  -> a dict of biomechanically meaningful angles
     (knees, hips, elbows, shoulders, torso incline, etc.) used directly by
     the rule-based FormChecker and RepCounter.

  2. `extract_feature_vector` -> a fixed-length, scale/translation invariant
     numeric vector used as input to the ML exercise classifier.
"""

from typing import Dict
import numpy as np

from .pose_estimator import LandmarkFrame, compute_angle, vertical_incline, euclidean

# Minimum average visibility for the landmarks used by an exercise before we
# trust the derived angles. Below this, we flag "person not fully visible".
VISIBILITY_THRESHOLD = 0.5

FEATURE_NAMES = [
    "elbow_L", "elbow_R",
    "shoulder_L", "shoulder_R",
    "hip_L", "hip_R",
    "knee_L", "knee_R",
    "torso_incline",
    "hip_width_ratio",
    "hand_to_shoulder_L", "hand_to_shoulder_R",
    "ankle_width_ratio",
    "wrist_height_rel_shoulder_L", "wrist_height_rel_shoulder_R",
]


def compute_joint_angles(lf: LandmarkFrame) -> Dict[str, float]:
    """Return a dictionary of all joint angles (degrees) used across the
    exercise rule set. Angles are computed per side (L/R) since a user may
    be filmed from an angle where one side is more reliable than the other.
    """
    g = lf.get  # shorthand

    angles = {
        # Elbow flexion: shoulder-elbow-wrist
        "elbow_L": compute_angle(g("LEFT_SHOULDER"), g("LEFT_ELBOW"), g("LEFT_WRIST")),
        "elbow_R": compute_angle(g("RIGHT_SHOULDER"), g("RIGHT_ELBOW"), g("RIGHT_WRIST")),

        # Shoulder flexion: elbow-shoulder-hip
        "shoulder_L": compute_angle(g("LEFT_ELBOW"), g("LEFT_SHOULDER"), g("LEFT_HIP")),
        "shoulder_R": compute_angle(g("RIGHT_ELBOW"), g("RIGHT_SHOULDER"), g("RIGHT_HIP")),

        # Hip flexion: shoulder-hip-knee
        "hip_L": compute_angle(g("LEFT_SHOULDER"), g("LEFT_HIP"), g("LEFT_KNEE")),
        "hip_R": compute_angle(g("RIGHT_SHOULDER"), g("RIGHT_HIP"), g("RIGHT_KNEE")),

        # Knee flexion: hip-knee-ankle
        "knee_L": compute_angle(g("LEFT_HIP"), g("LEFT_KNEE"), g("LEFT_ANKLE")),
        "knee_R": compute_angle(g("RIGHT_HIP"), g("RIGHT_KNEE"), g("RIGHT_ANKLE")),

        # Body-line straightness (used for plank / push-up back sag/pike):
        # shoulder-hip-ankle should be close to 180 degrees when flat.
        "body_line_L": compute_angle(g("LEFT_SHOULDER"), g("LEFT_HIP"), g("LEFT_ANKLE")),
        "body_line_R": compute_angle(g("RIGHT_SHOULDER"), g("RIGHT_HIP"), g("RIGHT_ANKLE")),

        # Torso incline relative to vertical (0 = upright, 90 = horizontal)
        "torso_incline": vertical_incline(
            (g("LEFT_SHOULDER") + g("RIGHT_SHOULDER")) / 2,
            (g("LEFT_HIP") + g("RIGHT_HIP")) / 2,
        ),

        # Knee valgus proxy: horizontal offset of knee from the hip-ankle line,
        # normalized by hip width. Positive = knee caving inward.
        "knee_valgus_L": _knee_valgus(g("LEFT_HIP"), g("LEFT_KNEE"), g("LEFT_ANKLE")),
        "knee_valgus_R": _knee_valgus(g("RIGHT_HIP"), g("RIGHT_KNEE"), g("RIGHT_ANKLE")),
    }
    return angles


def _knee_valgus(hip, knee, ankle) -> float:
    """Signed horizontal deviation of the knee from the hip-ankle line,
    normalized by hip-to-ankle vertical distance. ~0 = knee tracks over
    foot; large magnitude = knee caving in/out (valgus/varus)."""
    hip, knee, ankle = np.asarray(hip, dtype=np.float64), np.asarray(knee, dtype=np.float64), np.asarray(ankle, dtype=np.float64)
    vertical_span = abs(ankle[1] - hip[1]) + 1e-6
    # x position the knee "should" be at if hip-ankle were a straight vertical line
    t = (knee[1] - hip[1]) / (ankle[1] - hip[1] + 1e-6)
    expected_x = hip[0] + t * (ankle[0] - hip[0])
    return float((knee[0] - expected_x) / vertical_span)


def extract_feature_vector(lf: LandmarkFrame) -> np.ndarray:
    """Build a fixed-length feature vector for the ML exercise classifier.
    Distances are normalized by torso length so the vector is invariant to
    the person's distance from the camera and body size.
    """
    g = lf.get
    torso_len = euclidean(
        (g("LEFT_SHOULDER") + g("RIGHT_SHOULDER")) / 2,
        (g("LEFT_HIP") + g("RIGHT_HIP")) / 2,
    ) + 1e-6

    angles = compute_joint_angles(lf)

    hip_width = euclidean(g("LEFT_HIP"), g("RIGHT_HIP")) / torso_len
    ankle_width = euclidean(g("LEFT_ANKLE"), g("RIGHT_ANKLE")) / torso_len
    hand_to_shoulder_L = euclidean(g("LEFT_WRIST"), g("LEFT_SHOULDER")) / torso_len
    hand_to_shoulder_R = euclidean(g("RIGHT_WRIST"), g("RIGHT_SHOULDER")) / torso_len
    wrist_rel_L = (g("LEFT_SHOULDER")[1] - g("LEFT_WRIST")[1]) / torso_len
    wrist_rel_R = (g("RIGHT_SHOULDER")[1] - g("RIGHT_WRIST")[1]) / torso_len

    vec = np.array([
        angles["elbow_L"], angles["elbow_R"],
        angles["shoulder_L"], angles["shoulder_R"],
        angles["hip_L"], angles["hip_R"],
        angles["knee_L"], angles["knee_R"],
        angles["torso_incline"],
        hip_width,
        hand_to_shoulder_L, hand_to_shoulder_R,
        ankle_width,
        wrist_rel_L, wrist_rel_R,
    ], dtype=np.float32)

    return vec


def is_fully_visible(lf: LandmarkFrame, exercise: str = None) -> bool:
    """Sanity check that the key limbs are visible enough to trust the
    derived angles before we act on them."""
    core = [
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
    ]
    return lf.mean_visibility(core) >= VISIBILITY_THRESHOLD

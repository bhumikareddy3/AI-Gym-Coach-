"""
tests/helpers.py
-----------------
Synthetic LandmarkFrame builder so tests exercise pose_confidence,
temporal_analyzer, and form_analysis without running MediaPipe inference
on an actual image. All 33 landmarks are placed at plausible normalized
(x, y) positions for a person standing facing the camera; callers can
override individual landmark positions/visibility to simulate a squat
bottom, low confidence, occlusion, etc.
"""

from typing import Dict, Optional
import numpy as np

from app.pose_estimator import LandmarkFrame, PoseLandmark

# A rough, anatomically-plausible standing pose in normalized [0,1] image
# space (x grows right, y grows down), used as the default for every
# landmark not explicitly overridden.
_DEFAULT_POSITIONS: Dict[str, tuple] = {
    "NOSE": (0.50, 0.12),
    "LEFT_SHOULDER": (0.42, 0.25), "RIGHT_SHOULDER": (0.58, 0.25),
    "LEFT_ELBOW": (0.40, 0.40), "RIGHT_ELBOW": (0.60, 0.40),
    "LEFT_WRIST": (0.38, 0.55), "RIGHT_WRIST": (0.62, 0.55),
    "LEFT_HIP": (0.45, 0.55), "RIGHT_HIP": (0.55, 0.55),
    "LEFT_KNEE": (0.45, 0.75), "RIGHT_KNEE": (0.55, 0.75),
    "LEFT_ANKLE": (0.45, 0.95), "RIGHT_ANKLE": (0.55, 0.95),
}

IMG_W, IMG_H = 640, 480


def make_landmark_frame(
    overrides: Optional[Dict[str, tuple]] = None,
    visibility: Optional[Dict[str, float]] = None,
    default_visibility: float = 0.95,
) -> LandmarkFrame:
    """Build a synthetic LandmarkFrame. `overrides` sets normalized (x, y)
    for named landmarks (others keep a standing-pose default position of
    0.5, 0.5 if not in _DEFAULT_POSITIONS). `visibility` sets per-landmark
    visibility (0-1); unspecified landmarks get `default_visibility`."""
    overrides = overrides or {}
    visibility = visibility or {}

    n = len(PoseLandmark)
    norm = np.full((n, 3), 0.0, dtype=np.float32)
    vis = np.full((n,), default_visibility, dtype=np.float32)

    for lm in PoseLandmark:
        name = lm.name
        x, y = _DEFAULT_POSITIONS.get(name, (0.5, 0.5))
        if name in overrides:
            x, y = overrides[name]
        norm[lm.value] = [x, y, 0.0]
        if name in visibility:
            vis[lm.value] = visibility[name]

    px = norm[:, :2] * np.array([IMG_W, IMG_H], dtype=np.float32)
    return LandmarkFrame(norm=norm, px=px, visibility=vis, raw_results=None)


def squat_angles(depth: str = "top") -> Dict[str, float]:
    """Directly fabricate an angles dict (bypassing landmark geometry) for
    rep_counter / temporal_analyzer tests that only need the driving
    angle, not full pose geometry."""
    knee = 170.0 if depth == "top" else 85.0
    return {
        "knee_L": knee, "knee_R": knee,
        "hip_L": 170.0, "hip_R": 170.0,
        "elbow_L": 170.0, "elbow_R": 170.0,
        "torso_incline": 10.0,
        "body_line_L": 175.0, "body_line_R": 175.0,
    }

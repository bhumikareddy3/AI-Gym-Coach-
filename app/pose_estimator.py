"""
pose_estimator.py
-----------------
Wraps MediaPipe Pose to extract 33 body landmarks per frame, exposes helper
utilities for landmark lookup, joint-angle computation, and skeleton drawing.

This is the lowest layer of the stack -- every other module (feature
engineering, rep counting, form checking) consumes the LandmarkFrame object
produced here.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

PoseLandmark = mp_pose.PoseLandmark


@dataclass
class LandmarkFrame:
    """Container for a single frame's pose landmarks in normalized and
    pixel coordinates, plus per-landmark visibility (confidence)."""
    norm: np.ndarray        # (33, 3) x,y,z in [0,1] normalized image space
    px: np.ndarray          # (33, 2) x,y in pixel space
    visibility: np.ndarray  # (33,) confidence per landmark
    raw_results: object      # original mediapipe result (for drawing)

    def get(self, name: str) -> np.ndarray:
        """Return pixel-space (x, y) for a named landmark, e.g. 'LEFT_KNEE'."""
        idx = PoseLandmark[name].value
        return self.px[idx]

    def vis(self, name: str) -> float:
        idx = PoseLandmark[name].value
        return float(self.visibility[idx])

    def mean_visibility(self, names) -> float:
        return float(np.mean([self.vis(n) for n in names]))


class PoseEstimator:
    """Thin, stateful wrapper around mediapipe.solutions.pose.Pose tuned for
    real-time video (as opposed to static image) inference."""

    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        model_complexity: int = 1,
    ):
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[LandmarkFrame]:
        """Run pose inference on a BGR frame. Returns None if no person
        is detected in the frame."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._pose.process(frame_rgb)

        if not results.pose_landmarks:
            return None

        lms = results.pose_landmarks.landmark
        norm = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
        px = norm[:, :2] * np.array([w, h], dtype=np.float32)
        visibility = np.array([lm.visibility for lm in lms], dtype=np.float32)

        return LandmarkFrame(norm=norm, px=px, visibility=visibility, raw_results=results)

    def draw(self, frame_bgr: np.ndarray, lf: LandmarkFrame) -> np.ndarray:
        """Draw the skeleton overlay onto a copy of the frame."""
        annotated = frame_bgr.copy()
        mp_drawing.draw_landmarks(
            annotated,
            lf.raw_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
        )
        return annotated

    def close(self):
        self._pose.close()


def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the interior angle (in degrees) at vertex `b` formed by points
    a-b-c, using 2D (x, y) pixel or normalized coordinates.

    Example: compute_angle(hip, knee, ankle) -> knee flexion angle.
    """
    a, b, c = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), np.asarray(c, dtype=np.float64)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def vertical_incline(top: np.ndarray, bottom: np.ndarray) -> float:
    """
    Angle (degrees) between the segment top->bottom and the true vertical
    axis. 0 degrees = perfectly upright/vertical. Used for torso/back lean.
    """
    top, bottom = np.asarray(top, dtype=np.float64), np.asarray(bottom, dtype=np.float64)
    seg = top - bottom
    vertical = np.array([0.0, -1.0])  # image y-axis grows downward
    norm_seg = np.linalg.norm(seg)
    if norm_seg < 1e-6:
        return 0.0
    cosine = np.dot(seg, vertical) / norm_seg
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

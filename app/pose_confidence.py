"""
pose_confidence.py
-------------------
Turns per-landmark MediaPipe visibility into a single, actionable
pose-confidence signal, per spec section 3:

    LOW CONFIDENCE
          |
    Do not provide unreliable form correction
          |
    Ask user to reposition / improve camera visibility

`is_fully_visible()` in feature_engineering.py is a hard boolean gate on a
fixed landmark set. This module sits alongside it and adds a *graded*
confidence score plus a human-readable reason, so the frontend/coach layer
can:

  - show a live confidence meter,
  - explain *why* confidence is low (occlusion vs. distance vs. framing),
  - and suppress form feedback (and any LLM coaching call) below threshold
    without just saying "low confidence" with no actionable next step.

Nothing here replaces the existing visibility gate in feature_engineering.py
-- it is additive and safe to ignore if a caller only wants the old boolean.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from .pose_estimator import LandmarkFrame

# Landmarks that matter for virtually every supported exercise. Weighted
# higher than peripheral landmarks (wrists/ankles) because a bad hip/knee
# read corrupts every downstream angle.
CORE_LANDMARKS = [
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
]

# Per-exercise supplementary landmarks whose visibility matters for that
# exercise's specific angles (e.g. elbows/wrists for push-ups and curls).
EXERCISE_LANDMARKS = {
    "squat": [],
    "lunge": [],
    "plank": ["LEFT_ELBOW", "RIGHT_ELBOW"],
    "pushup": ["LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST"],
    "bicep_curl": ["LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST"],
}

CORE_WEIGHT = 0.75
EXTRA_WEIGHT = 0.25

# Confidence bands. "reliable" is the only band in which form-correction
# feedback (and any LLM/RAG coaching call) should be generated.
LOW_THRESHOLD = 0.5
MEDIUM_THRESHOLD = 0.7


@dataclass
class PoseConfidence:
    score: float                 # 0.0-1.0 overall confidence
    level: str                   # "low" | "medium" | "high"
    reliable: bool                # True only when level != "low"
    reason: Optional[str] = None  # populated when not reliable
    message: Optional[str] = None  # user-facing guidance when not reliable
    per_landmark: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "level": self.level,
            "reliable": self.reliable,
            "reason": self.reason,
            "message": self.message,
        }


def _classify(score: float) -> str:
    if score < LOW_THRESHOLD:
        return "low"
    if score < MEDIUM_THRESHOLD:
        return "medium"
    return "high"


def _diagnose(lf: LandmarkFrame, landmarks: List[str]) -> str:
    """Best-effort human explanation for why confidence is low, based on
    which landmarks are weak and where they sit in the frame."""
    weak = [n for n in landmarks if lf.vis(n) < LOW_THRESHOLD]
    if not weak:
        return "Pose is partially unclear -- hold still for a moment."

    lower_body = {"LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"}
    upper_body = {"LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST"}

    weak_lower = any(n in lower_body for n in weak)
    weak_upper = any(n in upper_body for n in weak)

    # Off-frame heuristic: landmark x/y close to the image edges (0 or 1)
    edge_hits = 0
    for n in weak:
        try:
            idx_xy = lf.norm[_idx(n)][:2]
        except Exception:
            continue
        if np.any(idx_xy < 0.03) or np.any(idx_xy > 0.97):
            edge_hits += 1

    if edge_hits >= max(1, len(weak) // 2):
        return "Part of your body is out of frame."
    if weak_lower and not weak_upper:
        return "Your lower body isn't clearly visible."
    if weak_upper and not weak_lower:
        return "Your upper body isn't clearly visible."
    return "Your full body isn't clearly visible."


def _idx(name: str) -> int:
    from .pose_estimator import PoseLandmark
    return PoseLandmark[name].value


def _message_for(level: str, reason: str) -> str:
    if level != "low":
        return None
    hints = {
        "Part of your body is out of frame.": "Please step back so your whole body is in frame.",
        "Your lower body isn't clearly visible.": "I can't clearly see your legs. Please move back or clear the area around your lower body.",
        "Your upper body isn't clearly visible.": "I can't clearly see your upper body. Please adjust the camera or your position.",
        "Your full body isn't clearly visible.": "I can't clearly see your full body. Please move farther from the camera.",
    }
    return hints.get(reason, "I can't get a reliable read on your pose right now -- please adjust your position or lighting.")


def compute_pose_confidence(lf: LandmarkFrame, exercise: Optional[str] = None) -> PoseConfidence:
    """Compute a graded pose-confidence score (0-1) for the current frame.

    Combines mean visibility of core landmarks (weighted higher) with any
    exercise-specific landmarks. Below LOW_THRESHOLD, `reliable` is False
    and `message`/`reason` explain what to fix -- callers (api.py, the
    LLM coach) should treat that as a hard gate: don't generate form
    correction or coaching text from an unreliable read.
    """
    extra = EXERCISE_LANDMARKS.get(exercise, []) if exercise else []
    landmarks = CORE_LANDMARKS + extra

    per_landmark = {n: float(lf.vis(n)) for n in landmarks}

    core_score = float(np.mean([per_landmark[n] for n in CORE_LANDMARKS]))
    if extra:
        extra_score = float(np.mean([per_landmark[n] for n in extra]))
        score = CORE_WEIGHT * core_score + EXTRA_WEIGHT * extra_score
    else:
        score = core_score

    score = float(np.clip(score, 0.0, 1.0))
    level = _classify(score)
    reliable = level != "low"

    reason = None
    message = None
    if not reliable:
        reason = _diagnose(lf, landmarks)
        message = _message_for(level, reason)

    return PoseConfidence(
        score=score, level=level, reliable=reliable,
        reason=reason, message=message, per_landmark=per_landmark,
    )

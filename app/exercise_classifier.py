"""
exercise_classifier.py
------------------------
Skeleton-based action recognition: classifies which exercise is being
performed from a single frame's engineered pose-angle feature vector.

Model: scikit-learn RandomForestClassifier over the 15-dim feature vector
from feature_engineering.extract_feature_vector. Random forests are a
deliberate choice here (over a deep net) because:
  - the feature vector is already hand-engineered and low-dimensional,
  - training data per user session is small,
  - it gives fast CPU-only real-time inference with no extra runtime deps.

A `predict_smoothed` majority-vote wrapper is provided so the live
classification is stable frame-to-frame rather than flickering.

If no trained model file is present, `RuleBasedFallbackClassifier` provides
a cold-start classifier using simple angle heuristics so the system is
usable before any training data has been collected.
"""

from collections import deque, Counter
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import joblib

from .feature_engineering import extract_feature_vector, compute_joint_angles, FEATURE_NAMES
from .exercise_rules import SUPPORTED_EXERCISES

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "exercise_classifier.joblib"


class RuleBasedFallbackClassifier:
    """Zero-training heuristic classifier based on gross body posture.
    Used only when no trained ML model is available yet."""

    def predict(self, angles: dict) -> Tuple[str, float]:
        torso = angles["torso_incline"]
        knee = min(angles["knee_L"], angles["knee_R"])
        elbow = min(angles["elbow_L"], angles["elbow_R"])
        body_line = min(angles["body_line_L"], angles["body_line_R"])

        # Horizontal torso + straight body-line => plank or push-up family
        if torso > 60:
            if elbow < 150:
                return "pushup", 0.55
            return "plank", 0.55

        # Mostly upright torso: distinguish by which joints are actively bending
        knee_bend = knee < 150
        elbow_bend = elbow < 150
        if knee_bend and not elbow_bend:
            return "squat", 0.55
        if elbow_bend and not knee_bend:
            return "bicep_curl", 0.5
        if knee_bend and elbow_bend:
            return "lunge", 0.4
        return "squat", 0.3  # default guess


class ExerciseClassifier:
    """Loads a trained model if present; otherwise falls back to heuristics.
    Includes temporal smoothing via majority vote over a short window."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, smoothing_window: int = 15):
        self.model_path = Path(model_path)
        self.model = None
        self.label_encoder_classes_ = None
        self._history = deque(maxlen=smoothing_window)
        self._fallback = RuleBasedFallbackClassifier()
        self._load()

    def _load(self):
        if self.model_path.exists():
            bundle = joblib.load(self.model_path)
            self.model = bundle["model"]
            self.label_encoder_classes_ = bundle["classes"]

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def predict_frame(self, lf) -> Tuple[str, float]:
        """Predict exercise for a single LandmarkFrame (no smoothing)."""
        if self.is_trained:
            vec = extract_feature_vector(lf).reshape(1, -1)
            proba = self.model.predict_proba(vec)[0]
            idx = int(np.argmax(proba))
            return self.label_encoder_classes_[idx], float(proba[idx])
        else:
            angles = compute_joint_angles(lf)
            return self._fallback.predict(angles)

    def predict_smoothed(self, lf) -> Tuple[str, float]:
        """Predict with temporal majority-vote smoothing to prevent
        frame-to-frame label flicker. Confidence is the vote share times
        the mean confidence of votes for the winning label."""
        label, conf = self.predict_frame(lf)
        self._history.append((label, conf))

        votes = Counter(l for l, _ in self._history)
        winner, count = votes.most_common(1)[0]
        vote_share = count / len(self._history)
        mean_conf = float(np.mean([c for l, c in self._history if l == winner]))
        # confidence reflects both how consistent the vote is and how
        # confident the model was on the winning frames
        return winner, float(mean_conf * vote_share)

    def reset_smoothing(self):
        self._history.clear()

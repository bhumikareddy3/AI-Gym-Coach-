"""
feedback_engine.py
--------------------
Turns raw form-check results + rep-counter state into throttled,
human-readable coaching feedback with confidence scores. This is the layer
that decides *what to say and when* -- without throttling, the system would
shout "lower your hips" every single frame, which is useless and annoying.

Design:
  - Each FormIssue code has an independent cooldown so different mistakes
    can surface without one dominating the queue.
  - A rolling posture-quality score (EMA) smooths the frame-level score from
    exercise_rules so the on-screen confidence doesn't jitter.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

from .exercise_rules import FORM_CHECKERS, FormResult


@dataclass
class CoachingMessage:
    text: str
    severity: str
    timestamp: float


class FeedbackEngine:
    def __init__(self, exercise: str, cooldown_sec: float = 3.0, ema_alpha: float = 0.2):
        if exercise not in FORM_CHECKERS:
            raise ValueError(f"No form checker registered for '{exercise}'")
        self.exercise = exercise
        self.cooldown_sec = cooldown_sec
        self.ema_alpha = ema_alpha
        self._last_shown: Dict[str, float] = {}
        self._smoothed_score: Optional[float] = None
        self.message_log: List[CoachingMessage] = []
        self.recent_scores = deque(maxlen=300)

    def evaluate(self, angles: dict, **checker_kwargs) -> FormResult:
        """Run the exercise's form checker and update the smoothed score."""
        checker = FORM_CHECKERS[self.exercise]
        result: FormResult = checker(angles, **checker_kwargs)

        if self._smoothed_score is None:
            self._smoothed_score = result.score
        else:
            self._smoothed_score = (
                self.ema_alpha * result.score + (1 - self.ema_alpha) * self._smoothed_score
            )
        self.recent_scores.append(result.score)
        return result

    def get_messages(self, result: FormResult) -> List[CoachingMessage]:
        """Return the subset of issues that are due to be (re)shown, respecting
        per-issue cooldowns, and log them."""
        now = time.time()
        due: List[CoachingMessage] = []
        for issue in result.issues:
            last = self._last_shown.get(issue.code, 0)
            if now - last >= self.cooldown_sec:
                msg = CoachingMessage(text=issue.message, severity=issue.severity, timestamp=now)
                due.append(msg)
                self._last_shown[issue.code] = now
                self.message_log.append(msg)

        if not result.issues:
            # Positive reinforcement, throttled the same way under a shared key
            last = self._last_shown.get("_good_form", 0)
            if now - last >= self.cooldown_sec * 2:
                msg = CoachingMessage(text="Good form — keep it up!", severity="positive", timestamp=now)
                due.append(msg)
                self._last_shown["_good_form"] = now
        return due

    @property
    def smoothed_score(self) -> float:
        return float(self._smoothed_score) if self._smoothed_score is not None else 100.0

    @property
    def average_form_score(self) -> float:
        if not self.recent_scores:
            return 100.0
        return float(sum(self.recent_scores) / len(self.recent_scores))

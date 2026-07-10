"""
recommender.py
----------------
AI-based workout recommendation: analyzes a user's historical session
performance (from PerformanceLogger) and detected weaknesses to recommend
exercises, target rep counts, and difficulty adjustments.

This is implemented as an interpretable, rule-driven scoring system rather
than a black-box model -- for a coaching product, recommendations need to
be explainable ("your squat form accuracy dropped, so reps were reduced and
depth cues were prioritized") rather than an opaque score.
"""

from dataclasses import dataclass
from typing import List
import statistics

from .performance_logger import PerformanceLogger, SessionSummary

WEAKNESS_THRESHOLD = 75.0  # below this score (0-100), a metric is a "weakness"
IMPROVEMENT_THRESHOLD = 90.0  # above this, consider raising difficulty


@dataclass
class Recommendation:
    exercise: str
    target_reps: int
    target_sets: int
    difficulty: str  # "easier" | "maintain" | "harder"
    focus_cues: List[str]
    rationale: str


class WorkoutRecommender:
    def __init__(self, logger: PerformanceLogger):
        self.logger = logger

    def recommend(self, user_id: str, exercise: str, base_reps: int = 10, base_sets: int = 3) -> Recommendation:
        history = [s for s in self.logger.get_user_history(user_id, limit=10) if s.exercise == exercise]

        if not history:
            return Recommendation(
                exercise=exercise,
                target_reps=base_reps,
                target_sets=base_sets,
                difficulty="maintain",
                focus_cues=["Establish a baseline — focus on controlled, full-range reps."],
                rationale="No prior sessions found; starting at default difficulty.",
            )

        recent = history[:5]
        avg_form = statistics.mean(s.form_accuracy or 0 for s in recent)
        avg_rep_acc = statistics.mean(s.rep_accuracy or 0 for s in recent)
        avg_rom = statistics.mean(s.rom_pct or 0 for s in recent)
        avg_consistency = statistics.mean(s.consistency or 0 for s in recent)

        focus_cues = []
        rationale_parts = []

        if avg_form < WEAKNESS_THRESHOLD:
            focus_cues.append("Prioritize form over speed — slow down each rep.")
            rationale_parts.append(f"form accuracy averaged {avg_form:.0f}%")
        if avg_rom < WEAKNESS_THRESHOLD:
            focus_cues.append("Increase range of motion — go through the full movement.")
            rationale_parts.append(f"range of motion averaged {avg_rom:.0f}%")
        if avg_consistency < WEAKNESS_THRESHOLD:
            focus_cues.append("Keep a steady, even tempo across reps.")
            rationale_parts.append(f"consistency averaged {avg_consistency:.0f}%")

        # Difficulty logic: only progress if the fundamentals (form, rom) are solid
        if avg_form >= IMPROVEMENT_THRESHOLD and avg_rom >= IMPROVEMENT_THRESHOLD and avg_rep_acc >= IMPROVEMENT_THRESHOLD:
            difficulty = "harder"
            target_reps = base_reps + 4
            target_sets = base_sets
            rationale_parts.append("all core metrics are strong, progressing difficulty")
        elif avg_form < WEAKNESS_THRESHOLD or avg_rom < WEAKNESS_THRESHOLD:
            difficulty = "easier"
            target_reps = max(6, base_reps - 3)
            target_sets = base_sets
            rationale_parts.append("reducing volume to rebuild form quality")
        else:
            difficulty = "maintain"
            target_reps = base_reps
            target_sets = base_sets
            rationale_parts.append("metrics are stable, holding current difficulty")

        if not focus_cues:
            focus_cues.append("Solid execution — maintain current technique.")

        rationale = f"Based on the last {len(recent)} session(s): " + "; ".join(rationale_parts) + "."

        return Recommendation(
            exercise=exercise,
            target_reps=target_reps,
            target_sets=target_sets,
            difficulty=difficulty,
            focus_cues=focus_cues,
            rationale=rationale,
        )

    def weakest_exercise(self, user_id: str) -> str:
        """Across all exercises the user has done, return the one with the
        lowest average form accuracy (i.e. where coaching is most needed)."""
        history = self.logger.get_user_history(user_id, limit=50)
        if not history:
            return "squat"
        by_exercise = {}
        for s in history:
            by_exercise.setdefault(s.exercise, []).append(s.form_accuracy or 0)
        avg_by_exercise = {ex: statistics.mean(v) for ex, v in by_exercise.items()}
        return min(avg_by_exercise, key=avg_by_exercise.get)

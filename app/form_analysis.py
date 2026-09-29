"""
form_analysis.py
------------------
Per spec section 5: a single structured, deterministic snapshot of a
session's current state, combining outputs that already exist in separate
modules (pose_confidence, exercise_rules.FormResult, rep_counter.RepCounter
/ PlankTimer, temporal_analyzer.TemporalMetrics) into one schema that:

  - is safe to persist (performance_logger, Stage 3),
  - is safe to hand to the RAG/LLM coach layer (Stage 4-5) as *the* source
    of truth for what physically happened -- the LLM must not invent or
    override anything in here,
  - is safe to return directly from the API for a frontend dashboard.

This module does no analysis of its own; it only assembles results
produced by the existing CV/ML pipeline. Nothing here is ML/LLM-driven.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .exercise_rules import FormResult
from .pose_confidence import PoseConfidence
from .temporal_analyzer import TemporalMetrics


@dataclass
class FormIssueOut:
    type: str
    message: str
    severity: str       # "minor" | "major"
    confidence: float   # derived from pose_confidence.score at detection time

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "message": self.message,
            "severity": self.severity,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class FormAnalysis:
    exercise: str
    form_score: float
    pose_confidence: float
    range_of_motion: float
    rep_count: int
    tempo_sec: float
    movement_quality: str
    tempo_label: str
    issues: List[FormIssueOut] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "exercise": self.exercise,
            "form_score": round(self.form_score, 1),
            "pose_confidence": round(self.pose_confidence, 2),
            "range_of_motion": round(self.range_of_motion, 1),
            "rep_count": self.rep_count,
            "tempo": round(self.tempo_sec, 2),
            "movement_quality": self.movement_quality,
            "tempo_label": self.tempo_label,
            "issues": [i.to_dict() for i in self.issues],
        }


def build_form_analysis(
    exercise: str,
    form_result: FormResult,
    pose_conf: PoseConfidence,
    rep_count: int,
    range_of_motion_pct: float,
    avg_tempo_sec: float,
    temporal: Optional[TemporalMetrics] = None,
) -> FormAnalysis:
    """Assemble a FormAnalysis snapshot from already-computed pipeline
    outputs. Called once per processed frame (or once per completed rep --
    caller's choice) after the pose-confidence gate has passed."""
    issues = [
        FormIssueOut(
            type=issue.code,
            message=issue.message,
            severity=issue.severity,
            confidence=pose_conf.score,
        )
        for issue in form_result.issues
    ]
    return FormAnalysis(
        exercise=exercise,
        form_score=form_result.score,
        pose_confidence=pose_conf.score,
        range_of_motion=range_of_motion_pct,
        rep_count=rep_count,
        tempo_sec=avg_tempo_sec,
        movement_quality=temporal.movement_quality if temporal else "unknown",
        tempo_label=temporal.tempo_label if temporal else "unknown",
        issues=issues,
    )

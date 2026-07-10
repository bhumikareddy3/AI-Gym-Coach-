"""
exercise_rules.py
------------------
The biomechanical "knowledge base": ideal joint-angle ranges per exercise,
and the rule-based FormChecker that compares live angles against those
ranges to detect concrete form mistakes (rounded back, shallow squat,
incomplete push-up, knee misalignment, etc.).

Ranges are intentionally generous (population-level, not elite-athlete
mobility) and are meant to be tuned per user in a production deployment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

SUPPORTED_EXERCISES = ["squat", "pushup", "lunge", "plank", "bicep_curl"]


@dataclass
class FormIssue:
    code: str
    message: str
    severity: str  # "minor" | "major"


@dataclass
class FormResult:
    score: float  # 0-100 posture quality confidence score
    issues: List[FormIssue] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.issues) == 0


# ---------------------------------------------------------------------------
# Ideal ranges. Each entry: (good_min, good_max) in degrees unless noted.
# ---------------------------------------------------------------------------
IDEAL_RANGES = {
    "squat": {
        "knee_bottom": (70, 100),      # knee angle at bottom of a "full" rep
        "knee_standing": (160, 185),
        "torso_incline_max": 45,       # max forward lean before "rounded back"
        "knee_valgus_max": 0.18,       # normalized knee cave-in tolerance
    },
    "pushup": {
        "elbow_bottom": (70, 100),
        "elbow_top": (155, 185),
        "body_line_min": 160,          # shoulder-hip-ankle; below = sagging/piking
    },
    "lunge": {
        "front_knee_bottom": (80, 100),
        "back_knee_bottom": (80, 110),
        "torso_incline_max": 30,
    },
    "plank": {
        "body_line_min": 165,
        "body_line_max": 190,
        "hip_sag_body_line_min": 165,
    },
    "bicep_curl": {
        "elbow_extended": (150, 185),
        "elbow_flexed": (30, 60),
        "shoulder_swing_max": 25,      # shoulder angle should stay mostly still
    },
}


def _score_from_deviation(value: float, lo: float, hi: float, tolerance: float = 20.0) -> float:
    """Map a value's distance outside [lo, hi] to a 0-100 score. Inside the
    range -> 100. Outside -> decays linearly to 0 over `tolerance` degrees."""
    if lo <= value <= hi:
        return 100.0
    dist = lo - value if value < lo else value - hi
    return max(0.0, 100.0 * (1 - dist / tolerance))


def check_squat_form(angles: Dict[str, float], at_bottom: bool) -> FormResult:
    issues = []
    scores = []
    r = IDEAL_RANGES["squat"]

    knee = min(angles["knee_L"], angles["knee_R"])  # deepest side
    if at_bottom:
        lo, hi = r["knee_bottom"]
        scores.append(_score_from_deviation(knee, lo, hi))
        if knee > hi:
            issues.append(FormIssue("shallow_squat", "Lower your hips — go deeper for a full rep.", "major"))
    else:
        lo, hi = r["knee_standing"]
        scores.append(_score_from_deviation(knee, lo, hi))

    if angles["torso_incline"] > r["torso_incline_max"]:
        issues.append(FormIssue("rounded_back", "Keep your back straighter — chest up.", "major"))
        scores.append(_score_from_deviation(angles["torso_incline"], 0, r["torso_incline_max"]))

    valgus = max(abs(angles["knee_valgus_L"]), abs(angles["knee_valgus_R"]))
    if valgus > r["knee_valgus_max"]:
        issues.append(FormIssue("knee_valgus", "Align your knees with your feet — avoid caving inward.", "major"))
        scores.append(max(0.0, 100 * (1 - (valgus - r["knee_valgus_max"]) / 0.2)))

    return FormResult(score=float(min(scores) if scores else 100.0), issues=issues)


def check_pushup_form(angles: Dict[str, float], at_bottom: bool) -> FormResult:
    issues = []
    scores = []
    r = IDEAL_RANGES["pushup"]

    elbow = min(angles["elbow_L"], angles["elbow_R"])
    if at_bottom:
        lo, hi = r["elbow_bottom"]
        scores.append(_score_from_deviation(elbow, lo, hi))
    else:
        lo, hi = r["elbow_top"]
        scores.append(_score_from_deviation(elbow, lo, hi))
        if elbow < hi - 15:
            issues.append(FormIssue("incomplete_pushup", "Extend your arms fully at the top.", "major"))

    body_line = min(angles["body_line_L"], angles["body_line_R"])
    if body_line < r["body_line_min"]:
        issues.append(FormIssue("hip_sag", "Keep your hips in line — avoid sagging or piking.", "major"))
        scores.append(_score_from_deviation(body_line, r["body_line_min"], 180, tolerance=25))

    return FormResult(score=float(min(scores) if scores else 100.0), issues=issues)


def check_lunge_form(angles: Dict[str, float], at_bottom: bool) -> FormResult:
    issues = []
    scores = []
    r = IDEAL_RANGES["lunge"]

    front_knee = min(angles["knee_L"], angles["knee_R"])
    if at_bottom:
        lo, hi = r["front_knee_bottom"]
        scores.append(_score_from_deviation(front_knee, lo, hi))
        if front_knee < lo - 10:
            issues.append(FormIssue("knee_over_toe", "Front knee is going too far forward — keep it above the ankle.", "major"))

    if angles["torso_incline"] > r["torso_incline_max"]:
        issues.append(FormIssue("leaning_forward", "Keep your torso upright.", "minor"))
        scores.append(_score_from_deviation(angles["torso_incline"], 0, r["torso_incline_max"]))

    return FormResult(score=float(min(scores) if scores else 100.0), issues=issues)


def check_plank_form(angles: Dict[str, float]) -> FormResult:
    issues = []
    r = IDEAL_RANGES["plank"]
    body_line = min(angles["body_line_L"], angles["body_line_R"])
    score = _score_from_deviation(body_line, r["body_line_min"], r["body_line_max"], tolerance=25)

    if body_line < r["hip_sag_body_line_min"]:
        issues.append(FormIssue("hip_sag", "Raise your hips — keep your body in a straight line.", "major"))

    return FormResult(score=float(score), issues=issues)


def check_bicep_curl_form(angles: Dict[str, float], at_top: bool) -> FormResult:
    issues = []
    scores = []
    r = IDEAL_RANGES["bicep_curl"]

    elbow = min(angles["elbow_L"], angles["elbow_R"])
    if at_top:
        lo, hi = r["elbow_flexed"]
        scores.append(_score_from_deviation(elbow, lo, hi))
    else:
        lo, hi = r["elbow_extended"]
        scores.append(_score_from_deviation(elbow, lo, hi))
        if elbow < hi - 20:
            issues.append(FormIssue("incomplete_extension", "Extend your arms fully at the bottom.", "minor"))

    shoulder_swing = max(angles["shoulder_L"], angles["shoulder_R"])
    if shoulder_swing > r["shoulder_swing_max"]:
        issues.append(FormIssue("shoulder_swing", "Keep your upper arm still — avoid swinging your shoulder.", "minor"))
        scores.append(_score_from_deviation(shoulder_swing, 0, r["shoulder_swing_max"]))

    return FormResult(score=float(min(scores) if scores else 100.0), issues=issues)


FORM_CHECKERS = {
    "squat": check_squat_form,
    "pushup": check_pushup_form,
    "lunge": check_lunge_form,
    "plank": check_plank_form,
    "bicep_curl": check_bicep_curl_form,
}

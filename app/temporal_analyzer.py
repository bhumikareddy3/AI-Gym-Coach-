"""
temporal_analyzer.py
---------------------
Per spec section 4: treats an exercise as a sequence of movements rather
than scoring each frame in isolation. Maintains a short rolling window of
recent joint-angle frames per session and derives temporal features:

  - angular velocity / acceleration of the exercise's driving joint
  - a smoothness/jerk estimate (how "jumpy" the motion is)
  - a movement-quality label: controlled | uncontrolled | jerky
  - a tempo label: fast | normal | slow, relative to a per-exercise
    reference range (not a hard rule -- advisory only)

This module is purely additive: it sits *around* the existing RandomForest
classifier and rule-based FormChecker (rep_counter.py, exercise_rules.py),
consuming the same `angles` dict they already compute per frame. It does
not replace or duplicate rep counting, ROM scoring, or form-issue
detection -- those stay authoritative in rep_counter.py / exercise_rules.py.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

# Which angle "drives" the movement for each exercise -- same convention
# rep_counter.THRESHOLDS uses, kept separate so this module has no import
# dependency on rep_counter (it can also run for exercises without a
# RepCounter, e.g. plank's body-line angle).
DRIVING_ANGLE = {
    "squat": "knee",
    "pushup": "elbow",
    "lunge": "knee",
    "bicep_curl": "elbow",
    "plank": "body_line",
}

# Advisory reference ranges for full-rep duration (seconds), used only to
# label tempo as fast/normal/slow -- never to fail a rep or block feedback.
TEMPO_REFERENCE_SEC = {
    "squat": (1.5, 4.0),
    "pushup": (1.2, 3.5),
    "lunge": (1.5, 4.0),
    "bicep_curl": (1.5, 4.0),
}

# Angular velocity (deg/sec) above which movement is flagged "uncontrolled"
# for the driving joint of a rep-based exercise. Generous on purpose --
# this is meant to catch genuinely fast, momentum-driven reps, not normal
# tempo variation.
UNCONTROLLED_VELOCITY_DEG_PER_SEC = 220.0

# Jerk (rate of change of acceleration) above which motion is flagged
# "jerky" -- i.e. not smooth/controlled even if average speed is fine.
JERKY_THRESHOLD = 900.0

WINDOW_SECONDS = 2.0
MAX_SAMPLES = 60  # hard cap regardless of frame rate


@dataclass
class TemporalMetrics:
    driving_angle_key: Optional[str]
    angular_velocity_deg_s: float = 0.0
    angular_acceleration_deg_s2: float = 0.0
    jerk: float = 0.0
    movement_quality: str = "insufficient_data"  # controlled | uncontrolled | jerky | insufficient_data
    tempo_label: str = "unknown"                  # fast | normal | slow | unknown
    samples_in_window: int = 0

    def to_dict(self) -> dict:
        return {
            "driving_angle": self.driving_angle_key,
            "angular_velocity_deg_s": round(self.angular_velocity_deg_s, 1),
            "angular_acceleration_deg_s2": round(self.angular_acceleration_deg_s2, 1),
            "jerk": round(self.jerk, 1),
            "movement_quality": self.movement_quality,
            "tempo_label": self.tempo_label,
        }


@dataclass
class _Sample:
    t: float
    angle: float


class TemporalAnalyzer:
    """One instance per active (session_id, exercise). Feed it the driving
    angle for that exercise once per processed frame via `update()`."""

    def __init__(self, exercise: str, window_seconds: float = WINDOW_SECONDS,
                 max_samples: int = MAX_SAMPLES):
        self.exercise = exercise
        self.angle_key = DRIVING_ANGLE.get(exercise)
        self.window_seconds = window_seconds
        self._buf: Deque[_Sample] = deque(maxlen=max_samples)
        self._last_velocity: Optional[float] = None

    def _driving_angle_value(self, angles: Dict[str, float]) -> Optional[float]:
        if self.angle_key is None:
            return None
        if self.angle_key == "body_line":
            # plank: use the more reliable (lower-magnitude deviation from
            # 180) side rather than a fixed L/R pick
            l, r = angles.get("body_line_L"), angles.get("body_line_R")
            vals = [v for v in (l, r) if v is not None]
            return min(vals, key=lambda v: abs(180 - v)) if vals else None
        l = angles.get(f"{self.angle_key}_L")
        r = angles.get(f"{self.angle_key}_R")
        vals = [v for v in (l, r) if v is not None]
        if not vals:
            return None
        # same convention as RepCounter: the more pronounced (smaller) side
        return min(vals)

    def update(self, angles: Dict[str, float], timestamp: Optional[float] = None) -> TemporalMetrics:
        angle = self._driving_angle_value(angles)
        now = timestamp if timestamp is not None else time.time()

        if angle is None:
            return TemporalMetrics(driving_angle_key=self.angle_key, movement_quality="insufficient_data")

        self._buf.append(_Sample(t=now, angle=angle))
        self._trim_window(now)

        if len(self._buf) < 3:
            return TemporalMetrics(
                driving_angle_key=self.angle_key,
                samples_in_window=len(self._buf),
                movement_quality="insufficient_data",
            )

        samples = list(self._buf)
        velocity = self._instant_velocity(samples[-2], samples[-1])
        prev_velocity = self._last_velocity if self._last_velocity is not None else velocity
        dt = max(samples[-1].t - samples[-2].t, 1e-3)
        acceleration = (velocity - prev_velocity) / dt
        jerk = abs(acceleration - self._prev_accel(samples)) / dt if len(samples) >= 4 else 0.0
        self._last_velocity = velocity

        quality = self._classify_quality(velocity, jerk)
        tempo = self._classify_tempo()

        return TemporalMetrics(
            driving_angle_key=self.angle_key,
            angular_velocity_deg_s=velocity,
            angular_acceleration_deg_s2=acceleration,
            jerk=jerk,
            movement_quality=quality,
            tempo_label=tempo,
            samples_in_window=len(samples),
        )

    def _trim_window(self, now: float):
        while self._buf and (now - self._buf[0].t) > self.window_seconds:
            self._buf.popleft()

    @staticmethod
    def _instant_velocity(a: "_Sample", b: "_Sample") -> float:
        dt = max(b.t - a.t, 1e-3)
        return (b.angle - a.angle) / dt

    def _prev_accel(self, samples: List["_Sample"]) -> float:
        a, b, c = samples[-4], samples[-3], samples[-2]
        v1 = self._instant_velocity(a, b)
        v2 = self._instant_velocity(b, c)
        dt = max(c.t - b.t, 1e-3)
        return (v2 - v1) / dt

    def _classify_quality(self, velocity: float, jerk: float) -> str:
        speed = abs(velocity)
        if jerk >= JERKY_THRESHOLD:
            return "jerky"
        if speed >= UNCONTROLLED_VELOCITY_DEG_PER_SEC:
            return "uncontrolled"
        return "controlled"

    def _classify_tempo(self) -> str:
        ref = TEMPO_REFERENCE_SEC.get(self.exercise)
        if not ref or len(self._buf) < 2:
            return "unknown"
        span = self._buf[-1].t - self._buf[0].t
        if span <= 0:
            return "unknown"
        # crude proxy: how much window time it took to cover the observed
        # angle swing, extrapolated to a full rep-like excursion
        angle_swing = max(s.angle for s in self._buf) - min(s.angle for s in self._buf)
        if angle_swing < 15:  # not enough motion yet to judge tempo
            return "unknown"
        low, high = ref
        if span < low * 0.5:
            return "fast"
        if span > high * 1.5:
            return "slow"
        return "normal"

    def reset(self):
        self._buf.clear()
        self._last_velocity = None

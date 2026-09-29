"""
rep_counter.py
--------------
Automatic repetition counting using joint-angle trajectories and finite
state-machine transitions (UP <-> DOWN) with hysteresis thresholds to avoid
double-counting on noisy frames. Also tracks per-rep range of motion (ROM)
so the FeedbackEngine can score how "complete" each rep was.

Each exercise gets its own threshold pair (enter_down, enter_up) rather than
a single midpoint, which prevents oscillation-driven false counts when the
angle hovers near a single cutoff.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RepRecord:
    rep_number: int
    min_angle: float
    max_angle: float
    duration_sec: float
    depth_ok: bool  # whether ROM reached the "good" threshold
    eccentric_sec: float = 0.0   # top -> bottom (lengthening) phase duration
    concentric_sec: float = 0.0  # bottom -> top (shortening) phase duration


# angle_key: which joint-angle drives the state machine for each exercise
# enter_down / enter_up: hysteresis thresholds (degrees)
# good_rom: the angle a rep's minimum must reach to count as "full depth"
THRESHOLDS = {
    "squat":      dict(angle_key="knee",   enter_down=110, enter_up=155, good_rom=100),
    "pushup":     dict(angle_key="elbow",  enter_down=110, enter_up=155, good_rom=100),
    "lunge":      dict(angle_key="knee",   enter_down=115, enter_up=155, good_rom=105),
    "bicep_curl": dict(angle_key="elbow",  enter_down=70,  enter_up=140, good_rom=60),
}


class RepCounter:
    """Stateful per-exercise repetition counter. Call `update(angles)` once
    per processed frame; it returns a RepRecord when a rep completes, or
    None otherwise."""

    def __init__(self, exercise: str):
        if exercise not in THRESHOLDS:
            raise ValueError(f"RepCounter does not support '{exercise}' (it is not rep-based).")
        self.exercise = exercise
        self.cfg = THRESHOLDS[exercise]
        self.state = "up"  # "up" = top/extended position, "down" = bottom/flexed
        self.count = 0
        self.current_min = 999.0
        self.current_max = -999.0
        self._rep_start_time: Optional[float] = None
        self._down_start_time: Optional[float] = None  # set when state -> "down"
        self.history: List[RepRecord] = []

    def _driving_angle(self, angles: dict) -> float:
        key = self.cfg["angle_key"]
        # use the side with the more pronounced (smaller/larger as relevant) angle
        return min(angles[f"{key}_L"], angles[f"{key}_R"])

    def update(self, angles: dict) -> Optional[RepRecord]:
        angle = self._driving_angle(angles)
        self.current_min = min(self.current_min, angle)
        self.current_max = max(self.current_max, angle)
        if self._rep_start_time is None:
            self._rep_start_time = time.time()

        completed: Optional[RepRecord] = None

        now = time.time()

        if self.state == "up" and angle <= self.cfg["enter_down"]:
            self.state = "down"
            self._down_start_time = now

        elif self.state == "down" and angle >= self.cfg["enter_up"]:
            # rep completed: went down then back up
            self.state = "up"
            self.count += 1
            duration = now - self._rep_start_time if self._rep_start_time is not None else 0.0
            eccentric_sec = (self._down_start_time - self._rep_start_time) if (
                self._down_start_time is not None and self._rep_start_time is not None) else 0.0
            concentric_sec = (now - self._down_start_time) if self._down_start_time is not None else 0.0
            depth_ok = self.current_min <= self.cfg["good_rom"]
            completed = RepRecord(
                rep_number=self.count,
                min_angle=self.current_min,
                max_angle=self.current_max,
                duration_sec=duration,
                depth_ok=depth_ok,
                eccentric_sec=round(max(0.0, eccentric_sec), 3),
                concentric_sec=round(max(0.0, concentric_sec), 3),
            )
            self.history.append(completed)
            # reset tracking window for next rep
            self.current_min, self.current_max = 999.0, -999.0
            self._rep_start_time = now
            self._down_start_time = None

        return completed

    @property
    def is_at_bottom(self) -> bool:
        return self.state == "down"

    def range_of_motion_pct(self) -> float:
        """Average, across completed reps, of how close min_angle got to the
        ideal 'good_rom' target -- 100% = matched or exceeded target depth."""
        if not self.history:
            return 0.0
        pct = []
        for rep in self.history:
            target = self.cfg["good_rom"]
            # smaller angle = deeper; ratio capped at 100%
            pct.append(min(100.0, 100.0 * target / max(rep.min_angle, 1e-3)))
        return float(sum(pct) / len(pct))

    def rep_accuracy_pct(self) -> float:
        if not self.history:
            return 0.0
        good = sum(1 for r in self.history if r.depth_ok)
        return 100.0 * good / len(self.history)

    def consistency_score(self) -> float:
        """Lower variance in rep duration => higher consistency score (0-100)."""
        if len(self.history) < 2:
            return 100.0
        durations = [r.duration_sec for r in self.history]
        mean = sum(durations) / len(durations)
        if mean == 0:
            return 100.0
        variance = sum((d - mean) ** 2 for d in durations) / len(durations)
        cv = (variance ** 0.5) / mean  # coefficient of variation
        return float(max(0.0, 100.0 * (1 - min(cv, 1.0))))

    def avg_tempo_sec(self) -> float:
        """Average total rep duration (seconds) -- reported to the user/UI
        as 'tempo'."""
        if not self.history:
            return 0.0
        return float(sum(r.duration_sec for r in self.history) / len(self.history))

    def avg_phase_sec(self) -> Dict[str, float]:
        """Average eccentric/concentric phase duration across completed reps."""
        if not self.history:
            return {"eccentric_sec": 0.0, "concentric_sec": 0.0}
        n = len(self.history)
        return {
            "eccentric_sec": float(sum(r.eccentric_sec for r in self.history) / n),
            "concentric_sec": float(sum(r.concentric_sec for r in self.history) / n),
        }


class PlankTimer:
    """Plank/isometric holds aren't rep-based; track hold duration and
    "broken form" time instead."""

    def __init__(self):
        self.start_time = time.time()
        self.total_good_time = 0.0
        self.total_bad_time = 0.0
        self._last_tick = self.start_time

    def update(self, form_ok: bool):
        now = time.time()
        dt = now - self._last_tick
        self._last_tick = now
        if form_ok:
            self.total_good_time += dt
        else:
            self.total_bad_time += dt

    @property
    def total_time(self) -> float:
        return self.total_good_time + self.total_bad_time

    def completion_pct(self, target_seconds: float) -> float:
        return float(min(100.0, 100.0 * self.total_time / target_seconds)) if target_seconds > 0 else 0.0

    def form_accuracy_pct(self) -> float:
        if self.total_time == 0:
            return 100.0
        return float(100.0 * self.total_good_time / self.total_time)

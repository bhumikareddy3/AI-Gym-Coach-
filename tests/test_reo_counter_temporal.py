import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rep_counter import RepCounter
from tests.helpers import squat_angles


def _angles(knee):
    a = squat_angles()
    a["knee_L"] = a["knee_R"] = knee
    return a


def test_eccentric_concentric_phase_split():
    counter = RepCounter("squat")
    # time.time() call order: [rep_start_time init, now#1, now#2, now#3, now#4]
    # frame1 (angle=170): init rep_start_time=0.0, now=0.0  -> state stays "up"
    # frame2 (angle=100): now=1.0                             -> "up"->"down"
    # frame3 (angle=90):  now=1.5                              -> still "down"
    # frame4 (angle=160): now=3.0                              -> "down"->"up", rep completes
    with patch("app.rep_counter.time.time", side_effect=[0.0, 0.0, 1.0, 1.5, 3.0]):
        assert counter.update(_angles(170)) is None
        assert counter.update(_angles(100)) is None
        assert counter.update(_angles(90)) is None
        rep = counter.update(_angles(160))

    assert rep is not None
    assert rep.rep_number == 1
    assert rep.min_angle == 90
    assert rep.duration_sec == 3.0
    assert abs(rep.eccentric_sec - 1.0) < 1e-6   # top -> bottom took 1.0s
    assert abs(rep.concentric_sec - 2.0) < 1e-6  # bottom -> top took 2.0s


def test_avg_tempo_and_phase_helpers_empty_history():
    counter = RepCounter("squat")
    assert counter.avg_tempo_sec() == 0.0
    assert counter.avg_phase_sec() == {"eccentric_sec": 0.0, "concentric_sec": 0.0}


def test_avg_phase_sec_after_completed_reps():
    counter = RepCounter("squat")
    with patch("app.rep_counter.time.time", side_effect=[0.0, 0.0, 1.0, 3.0]):
        counter.update(_angles(170))
        counter.update(_angles(90))
        counter.update(_angles(160))
    avg = counter.avg_phase_sec()
    assert avg["eccentric_sec"] == 1.0
    assert avg["concentric_sec"] == 2.0
    assert counter.avg_tempo_sec() == 3.0

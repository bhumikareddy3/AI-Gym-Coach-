import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.temporal_analyzer import TemporalAnalyzer
from tests.helpers import squat_angles


def _feed(analyzer, angle_sequence, dt=0.05):
    """Feed a sequence of knee angles at a fixed simulated frame interval."""
    t = 0.0
    last = None
    for knee in angle_sequence:
        angles = squat_angles()
        angles["knee_L"] = angles["knee_R"] = knee
        last = analyzer.update(angles, timestamp=t)
        t += dt
    return last


def test_insufficient_data_for_first_frames():
    analyzer = TemporalAnalyzer("squat")
    m = analyzer.update(squat_angles(), timestamp=0.0)
    assert m.movement_quality == "insufficient_data"


def test_smooth_controlled_squat_descent():
    analyzer = TemporalAnalyzer("squat")
    # slow, steady descent from 170 -> 90 over 2 seconds (40 frames @ 50ms)
    seq = [170 - i * 2 for i in range(40)]
    m = _feed(analyzer, seq, dt=0.05)
    assert m.movement_quality in ("controlled",)
    assert m.samples_in_window > 0


def test_fast_jerky_motion_flagged_uncontrolled():
    analyzer = TemporalAnalyzer("squat")
    # violent oscillation: huge angle swings frame-to-frame
    seq = [170, 60, 170, 60, 170, 60, 170, 60]
    m = _feed(analyzer, seq, dt=0.03)
    assert m.movement_quality in ("uncontrolled", "jerky")


def test_plank_uses_body_line_angle():
    analyzer = TemporalAnalyzer("plank")
    angles = squat_angles()
    angles["body_line_L"] = 178.0
    angles["body_line_R"] = 176.0
    m = analyzer.update(angles, timestamp=0.0)
    assert m.driving_angle_key == "body_line"


def test_reset_clears_window():
    analyzer = TemporalAnalyzer("squat")
    _feed(analyzer, [170, 160, 150, 140], dt=0.05)
    assert len(analyzer._buf) > 0
    analyzer.reset()
    assert len(analyzer._buf) == 0

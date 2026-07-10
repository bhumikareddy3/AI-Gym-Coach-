"""
main.py
--------
Real-time AI Gym Coach: reads webcam frames, runs pose estimation, classifies
(or uses the user-selected) exercise, checks form, counts reps, generates
feedback, overlays everything on the video, and logs the session to SQLite
on exit.

Usage:
    python -m app.main --exercise squat --user_id bhumika
    python -m app.main --exercise auto --user_id bhumika       # ML-classified
    python -m app.main --exercise plank --user_id bhumika --target_seconds 60

Controls:
    q  -> quit and save session summary
    r  -> reset rep counter / plank timer
"""

import argparse
import time

import cv2

from .pose_estimator import PoseEstimator
from .feature_engineering import compute_joint_angles, is_fully_visible
from .exercise_classifier import ExerciseClassifier
from .rep_counter import RepCounter, PlankTimer, THRESHOLDS
from .feedback_engine import FeedbackEngine
from .performance_logger import PerformanceLogger
from .exercise_rules import SUPPORTED_EXERCISES

COLOR_OK = (80, 220, 80)
COLOR_WARN = (0, 165, 255)
COLOR_BAD = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)


def score_color(score: float):
    if score >= 85:
        return COLOR_OK
    if score >= 60:
        return COLOR_WARN
    return COLOR_BAD


def run(exercise_arg: str, user_id: str, camera_index: int, target_seconds: float):
    estimator = PoseEstimator()
    classifier = ExerciseClassifier() if exercise_arg == "auto" else None
    logger = PerformanceLogger()

    active_exercise = None if exercise_arg == "auto" else exercise_arg
    rep_counter = None
    plank_timer = None
    feedback_engine = None
    session_id = None

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    prev_time = time.time()
    on_screen_messages = []

    print("Starting AI Gym Coach. Press 'q' to quit, 'r' to reset counter.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            lf = estimator.process(frame)

            if lf is not None and is_fully_visible(lf):
                annotated = estimator.draw(frame, lf)

                # Determine exercise (auto-classify or fixed)
                if exercise_arg == "auto":
                    pred_exercise, conf = classifier.predict_smoothed(lf)
                    if pred_exercise != active_exercise and conf > 0.5:
                        active_exercise = pred_exercise
                        rep_counter = RepCounter(active_exercise) if active_exercise in THRESHOLDS else None
                        plank_timer = PlankTimer() if active_exercise == "plank" else None
                        feedback_engine = FeedbackEngine(active_exercise)
                        if session_id is not None:
                            _close_session(logger, session_id, rep_counter, plank_timer, target_seconds)
                        session_id = logger.start_session(user_id, active_exercise)

                if active_exercise is None:
                    cv2.putText(annotated, "Detecting exercise...", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
                else:
                    if feedback_engine is None:
                        feedback_engine = FeedbackEngine(active_exercise)
                    if session_id is None:
                        session_id = logger.start_session(user_id, active_exercise)
                    if active_exercise in THRESHOLDS and rep_counter is None:
                        rep_counter = RepCounter(active_exercise)
                    if active_exercise == "plank" and plank_timer is None:
                        plank_timer = PlankTimer()

                    angles = compute_joint_angles(lf)

                    if active_exercise == "plank":
                        result = feedback_engine.evaluate(angles)
                        plank_timer.update(form_ok=result.is_clean)
                        hold_text = f"Hold: {plank_timer.total_time:0.1f}s"
                        cv2.putText(annotated, hold_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, COLOR_TEXT, 2)
                    else:
                        at_extreme = rep_counter.is_at_bottom if rep_counter else False
                        checker_kwargs = {}
                        if active_exercise in ("squat", "lunge"):
                            checker_kwargs = {"at_bottom": at_extreme}
                        elif active_exercise == "pushup":
                            checker_kwargs = {"at_bottom": at_extreme}
                        elif active_exercise == "bicep_curl":
                            checker_kwargs = {"at_top": at_extreme}
                        result = feedback_engine.evaluate(angles, **checker_kwargs)

                        completed_rep = rep_counter.update(angles) if rep_counter else None
                        if completed_rep and session_id is not None:
                            logger.log_rep(
                                session_id, completed_rep.rep_number, completed_rep.min_angle,
                                completed_rep.max_angle, completed_rep.duration_sec, completed_rep.depth_ok,
                            )
                        rep_text = f"Reps: {rep_counter.count if rep_counter else 0}"
                        cv2.putText(annotated, rep_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, COLOR_TEXT, 2)

                    score = feedback_engine.smoothed_score
                    cv2.putText(annotated, f"Form: {score:0.0f}%", (20, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, score_color(score), 2)
                    cv2.putText(annotated, f"Exercise: {active_exercise}", (20, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

                    new_msgs = feedback_engine.get_messages(result)
                    if new_msgs:
                        on_screen_messages = new_msgs

                for i, msg in enumerate(on_screen_messages[-3:]):
                    color = COLOR_OK if msg.severity == "positive" else (COLOR_WARN if msg.severity == "minor" else COLOR_BAD)
                    cv2.putText(annotated, msg.text, (20, annotated.shape[0] - 20 - 30 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                annotated = frame
                cv2.putText(annotated, "Move into full view of the camera", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WARN, 2)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(annotated, f"{fps:0.0f} FPS", (annotated.shape[1] - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)

            cv2.imshow("AI Gym Coach", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                if rep_counter:
                    rep_counter = RepCounter(active_exercise)
                if plank_timer:
                    plank_timer = PlankTimer()

    finally:
        if session_id is not None:
            _close_session(logger, session_id, rep_counter, plank_timer, target_seconds, feedback_engine)
        cap.release()
        cv2.destroyAllWindows()
        estimator.close()


def _close_session(logger, session_id, rep_counter, plank_timer, target_seconds, feedback_engine=None):
    form_acc = feedback_engine.average_form_score if feedback_engine is not None else 100.0
    if plank_timer is not None:
        logger.end_session(
            session_id,
            form_accuracy=plank_timer.form_accuracy_pct(),
            rep_accuracy=100.0,
            rom_pct=100.0,
            consistency=100.0,
            completion_pct=plank_timer.completion_pct(target_seconds),
            total_reps=0,
        )
        print(f"Session saved. Plank hold: {plank_timer.total_time:0.1f}s, "
              f"form accuracy: {plank_timer.form_accuracy_pct():0.0f}%")
    elif rep_counter is not None:
        logger.end_session(
            session_id,
            form_accuracy=form_acc,
            rep_accuracy=rep_counter.rep_accuracy_pct(),
            rom_pct=rep_counter.range_of_motion_pct(),
            consistency=rep_counter.consistency_score(),
            completion_pct=min(100.0, 100.0 * rep_counter.count / 10),
            total_reps=rep_counter.count,
        )
        print(f"Session saved. Reps: {rep_counter.count}, "
              f"rep accuracy: {rep_counter.rep_accuracy_pct():0.0f}%, "
              f"ROM: {rep_counter.range_of_motion_pct():0.0f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time AI Gym Coach")
    parser.add_argument("--exercise", type=str, default="auto",
                         choices=["auto"] + SUPPORTED_EXERCISES,
                         help="Exercise to track, or 'auto' to ML-classify live")
    parser.add_argument("--user_id", type=str, default="default_user")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--target_seconds", type=float, default=60.0,
                         help="Target hold duration for plank completion %%")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.exercise, args.user_id, args.camera_index, args.target_seconds)

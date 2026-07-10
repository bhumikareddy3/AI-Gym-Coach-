"""
api.py
-------
FastAPI layer exposing model inference only (per the spec: "FastAPI only for
model inference" -- this is not a full webcam-streaming backend). A client
(mobile app, browser front-end, etc.) captures frames itself and POSTs each
one here; the server runs pose estimation + classification + form checking
+ rep counting and returns structured results. Per-session state (rep
counters, feedback engines) is kept in an in-memory dict keyed by session_id.

Run:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /session/start                 -> create a session, returns session_id
    POST /session/{id}/analyze_frame    -> analyze one base64 JPEG frame
    POST /session/{id}/end              -> finalize + persist session summary
    GET  /session/{id}/summary          -> retrieve stored summary
    GET  /users/{user_id}/recommendation?exercise=squat
"""

import base64
import uuid
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.pose_estimator import PoseEstimator, PoseLandmark
from app.feature_engineering import compute_joint_angles, is_fully_visible
from app.exercise_classifier import ExerciseClassifier
from app.rep_counter import RepCounter, PlankTimer, THRESHOLDS
from app.feedback_engine import FeedbackEngine
from app.performance_logger import PerformanceLogger
from app.recommender import WorkoutRecommender
from app.exercise_rules import SUPPORTED_EXERCISES

app = FastAPI(title="AI Gym Coach - Inference API", version="1.0.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(STATIC_DIR / "index.html")


# Keypoints + connections exposed to the browser for drawing the skeleton
# overlay. Kept to the subset feature_engineering already computes angles
# from, so no extra pose-estimation cost.
UI_KEYPOINTS = [
    "NOSE",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
]


def get_ui_landmarks(lf) -> Dict[str, list]:
    """Normalized (0-1) x,y per keypoint, ready for the browser to scale by
    canvas width/height and draw a skeleton overlay."""
    out = {}
    for name in UI_KEYPOINTS:
        idx = PoseLandmark[name].value
        x, y = lf.norm[idx][0], lf.norm[idx][1]
        out[name] = [round(float(x), 4), round(float(y), 4)]
    return out


_pose_estimator = PoseEstimator()
_logger = PerformanceLogger()
_recommender = WorkoutRecommender(_logger)

# In-memory per-session inference state (swap for Redis in a multi-worker deployment)
_sessions: Dict[str, dict] = {}


class StartSessionRequest(BaseModel):
    user_id: str
    exercise: str  # one of SUPPORTED_EXERCISES, or "auto"


class FrameRequest(BaseModel):
    image_base64: str  # JPEG/PNG bytes, base64-encoded, no data-URI prefix


class EndSessionRequest(BaseModel):
    target_seconds: Optional[float] = 60.0


@app.post("/session/start")
def start_session(req: StartSessionRequest):
    if req.exercise != "auto" and req.exercise not in SUPPORTED_EXERCISES:
        raise HTTPException(400, f"Unsupported exercise '{req.exercise}'")

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "user_id": req.user_id,
        "exercise_mode": req.exercise,   # "auto" or fixed name
        "active_exercise": None if req.exercise == "auto" else req.exercise,
        "classifier": ExerciseClassifier() if req.exercise == "auto" else None,
        "rep_counter": RepCounter(req.exercise) if req.exercise in THRESHOLDS else None,
        "plank_timer": PlankTimer() if req.exercise == "plank" else None,
        "feedback_engine": FeedbackEngine(req.exercise) if req.exercise in SUPPORTED_EXERCISES else None,
        "db_session_id": _logger.start_session(req.user_id, req.exercise) if req.exercise in SUPPORTED_EXERCISES else None,
    }
    return {"session_id": session_id}


def _decode_frame(image_base64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(image_base64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None")
        return frame
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")


@app.post("/session/{session_id}/analyze_frame")
def analyze_frame(session_id: str, req: FrameRequest):
    if session_id not in _sessions:
        raise HTTPException(404, "Unknown session_id")
    s = _sessions[session_id]

    frame = _decode_frame(req.image_base64)
    lf = _pose_estimator.process(frame)

    if lf is None or not is_fully_visible(lf):
        return {"person_detected": False, "message": "Move into full view of the camera"}

    # Resolve active exercise (auto-classify if needed)
    if s["exercise_mode"] == "auto":
        pred_exercise, conf = s["classifier"].predict_smoothed(lf)
        if pred_exercise != s["active_exercise"] and conf > 0.5:
            s["active_exercise"] = pred_exercise
            s["rep_counter"] = RepCounter(pred_exercise) if pred_exercise in THRESHOLDS else None
            s["plank_timer"] = PlankTimer() if pred_exercise == "plank" else None
            s["feedback_engine"] = FeedbackEngine(pred_exercise)
            if s["db_session_id"] is not None:
                _finalize_db_session(s, target_seconds=60.0)
            s["db_session_id"] = _logger.start_session(s["user_id"], pred_exercise)

    exercise = s["active_exercise"]
    if exercise is None:
        return {"person_detected": True, "active_exercise": None, "message": "Detecting exercise..."}

    angles = compute_joint_angles(lf)
    rep_counter: Optional[RepCounter] = s["rep_counter"]
    plank_timer: Optional[PlankTimer] = s["plank_timer"]
    fe: FeedbackEngine = s["feedback_engine"]

    response = {"person_detected": True, "active_exercise": exercise}

    if exercise == "plank":
        result = fe.evaluate(angles)
        plank_timer.update(form_ok=result.is_clean)
        response["hold_seconds"] = round(plank_timer.total_time, 2)
    else:
        at_extreme = rep_counter.is_at_bottom if rep_counter else False
        kwargs = {}
        if exercise in ("squat", "lunge", "pushup"):
            kwargs = {"at_bottom": at_extreme}
        elif exercise == "bicep_curl":
            kwargs = {"at_top": at_extreme}
        result = fe.evaluate(angles, **kwargs)

        completed_rep = rep_counter.update(angles) if rep_counter else None
        if completed_rep and s["db_session_id"] is not None:
            _logger.log_rep(s["db_session_id"], completed_rep.rep_number, completed_rep.min_angle,
                             completed_rep.max_angle, completed_rep.duration_sec, completed_rep.depth_ok)
        response["rep_count"] = rep_counter.count if rep_counter else 0

    messages = fe.get_messages(result)
    response["form_score"] = round(fe.smoothed_score, 1)
    response["issues"] = [{"code": i.code, "message": i.message, "severity": i.severity} for i in result.issues]
    response["feedback"] = [m.text for m in messages]
    response["joint_angles"] = {k: round(v, 1) for k, v in angles.items()}
    response["landmarks"] = get_ui_landmarks(lf)
    return response


def _finalize_db_session(s: dict, target_seconds: float):
    rep_counter: Optional[RepCounter] = s.get("rep_counter")
    plank_timer: Optional[PlankTimer] = s.get("plank_timer")
    fe: Optional[FeedbackEngine] = s.get("feedback_engine")
    form_acc = fe.average_form_score if fe else 100.0

    if plank_timer is not None:
        _logger.end_session(
            s["db_session_id"], form_accuracy=plank_timer.form_accuracy_pct(),
            rep_accuracy=100.0, rom_pct=100.0, consistency=100.0,
            completion_pct=plank_timer.completion_pct(target_seconds), total_reps=0,
        )
    elif rep_counter is not None:
        _logger.end_session(
            s["db_session_id"], form_accuracy=form_acc,
            rep_accuracy=rep_counter.rep_accuracy_pct(), rom_pct=rep_counter.range_of_motion_pct(),
            consistency=rep_counter.consistency_score(),
            completion_pct=min(100.0, 100.0 * rep_counter.count / 10), total_reps=rep_counter.count,
        )


@app.post("/session/{session_id}/end")
def end_session(session_id: str, req: EndSessionRequest = EndSessionRequest()):
    if session_id not in _sessions:
        raise HTTPException(404, "Unknown session_id")
    s = _sessions[session_id]
    if s["db_session_id"] is not None:
        _finalize_db_session(s, target_seconds=req.target_seconds)
    summary = _logger.get_session_summary(s["db_session_id"]) if s["db_session_id"] else None
    del _sessions[session_id]
    return {"summary": summary.__dict__ if summary else None}


@app.get("/session_db/{db_session_id}/summary")
def get_summary(db_session_id: int):
    summary = _logger.get_session_summary(db_session_id)
    if summary is None:
        raise HTTPException(404, "Unknown session")
    return summary.__dict__


@app.get("/users/{user_id}/recommendation")
def get_recommendation(user_id: str, exercise: str):
    if exercise not in SUPPORTED_EXERCISES:
        raise HTTPException(400, f"Unsupported exercise '{exercise}'")
    rec = _recommender.recommend(user_id, exercise)
    return rec.__dict__


@app.get("/health")
def health():
    return {"status": "ok"}

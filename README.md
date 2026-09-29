# AI Gym Coach — Real-Time Pose-Based Exercise Analysis

A real-time computer-vision fitness coach: webcam → MediaPipe pose landmarks →
joint-angle biomechanics → rule-based form checking + ML exercise
classification → rep counting → throttled natural-language feedback →
SQLite performance logging → rule-driven workout recommendations.

Frontend is intentionally minimal (an OpenCV overlay window). All engineering
effort is in the AI/CV pipeline, per the project brief.

# AI Gym Coach — Real-Time Pose-Based Exercise Analysis

A real-time computer-vision fitness coach: webcam → MediaPipe pose landmarks →
joint-angle biomechanics → rule-based form checking + ML exercise
classification → rep counting → throttled natural-language feedback →
SQLite performance logging → rule-driven workout recommendations.

Three ways to use it: a **browser UI** (easiest — no separate window, works
on any machine with a webcam and a browser), an **OpenCV desktop app**
(`main.py`), and a **plain inference API** (`api.py`) for building your own
frontend. All three share the exact same AI/CV pipeline underneath.

## Architecture

```
Webcam Frame
     │
     ▼
PoseEstimator (MediaPipe Pose, 33 landmarks)              ── app/pose_estimator.py
     │
     ▼
Feature Engineering                                        ── app/feature_engineering.py
  ├─ compute_joint_angles()  → knee/hip/elbow/shoulder/torso/valgus angles
  └─ extract_feature_vector() → 15-dim normalized vector for ML classifier
     │
     ├────────────────────────────┐
     ▼                            ▼
ExerciseClassifier          exercise_rules.py
(RandomForest, rule           (ideal biomechanical
 fallback if untrained)        ranges per exercise)     ── app/exercise_classifier.py
     │                            │                          app/exercise_rules.py
     └────────────┬───────────────┘
                  ▼
           FormChecker (per exercise)
           → FormResult(score, issues[])
                  │
     ┌────────────┼─────────────────┐
     ▼            ▼                 ▼
RepCounter /  FeedbackEngine    PerformanceLogger
PlankTimer    (EMA score,        (SQLite: sessions, reps) ── app/performance_logger.py
(state          throttled
 machine,       messages)     ── app/feedback_engine.py
 ROM,
 consistency) ── app/rep_counter.py
     │                                    │
     └──────────────┬─────────────────────┘
                    ▼
          WorkoutRecommender                              ── app/recommender.py
     (weakness detection, difficulty progression)
                    │
        ┌───────────┼─────────────────────┐
        ▼           ▼                     ▼
  main.py      static/ (browser UI)    api.py (FastAPI)
  OpenCV       webcam capture in-       serves the browser UI +
  overlay      browser, skeleton        POST /session/*/analyze_frame
               canvas overlay
```

See `docs/architecture.mmd` for the Mermaid source (renders on GitHub or
https://mermaid.live).

## Browser UI (recommended for demos)

The FastAPI server now also serves a small web frontend, so you can run
everything with one command and see it in a browser — no OpenCV window,
no separate client code.

```bash
uvicorn api:app --reload --port 8000
```

Then open **http://localhost:8000** in Chrome or Edge (needs webcam
permission). What it does:

- Captures your webcam feed directly in the browser (`getUserMedia`)
- Sends frames to `/session/{id}/analyze_frame` roughly 3x/second
- Draws the live skeleton overlay on a `<canvas>` using the returned joint
  landmarks
- Shows a large rep counter (or hold timer for planks), a form-score bar,
  and a live-scrolling feed of coaching messages
- The video frame's border glows lime/amber/coral in real time based on
  form score — so the frame itself is the at-a-glance indicator
- On "End session", shows the full performance summary (form accuracy, rep
  accuracy, ROM%, consistency, completion%) and a "Get recommendation"
  button that calls `/users/{id}/recommendation`

Frontend source: `static/index.html`, `static/style.css`, `static/app.js`
— plain HTML/CSS/JS, no build step, no framework.

**Note:** browser webcam access requires either `localhost` or HTTPS — if
you deploy this somewhere other than your own machine, you'll need a TLS
certificate for the camera prompt to work.

## Why these design choices

- **MediaPipe Pose** over a custom keypoint model: production-grade, runs
  CPU-only in real time, 33 landmarks is more than enough for the angles
  needed here.
- **Rule-based FormChecker as the primary correctness engine**: biomechanical
  "good squat depth" or "straight back" thresholds are well-established
  physiotherapy/strength-coaching knowledge — encoding them as rules is more
  reliable and *auditable* than expecting a small ML model to learn them from
  scratch, and it works from frame one with zero training data.
- **ML classifier for *which exercise*, not *is it correct***: exercise
  identification is a genuine pattern-recognition problem (which the
  RandomForest over engineered angle features handles well), whereas "is the
  knee too far forward" is a deterministic geometric check.
- **Hysteresis-based state machine for rep counting** instead of a single
  angle threshold: a single cutoff causes double-counting when the angle
  hovers near it; two thresholds (enter_down / enter_up) require a full
  down-then-up traversal before counting a rep.
- **SQLite** for persistence: zero-ops, file-based, sufficient for
  single-user/local deployments; swap for Postgres by only touching
  `performance_logger.py`.
- **Plain JS frontend, no framework**: the brief calls for minimal frontend
  emphasis — a build step (React/Vite/etc.) would add complexity without
  adding capability here. The frontend's only job is to show the AI
  pipeline's output; it deliberately stays out of the way.

## AI/ML pipeline detail

| Stage | Technique | File |
|---|---|---|
| Pose estimation | MediaPipe Pose (BlazePose backbone) | `pose_estimator.py` |
| Feature engineering | Joint-angle geometry, normalized distances | `feature_engineering.py` |
| Exercise recognition | RandomForestClassifier over 15-dim feature vector, temporal majority-vote smoothing | `exercise_classifier.py`, `train_classifier.py` |
| Posture analysis | Rule-based deviation scoring against biomechanical ranges | `exercise_rules.py` |
| Rep counting | Hysteresis finite-state machine over driving joint angle | `rep_counter.py` |
| Feedback | EMA-smoothed scoring + per-issue cooldown throttling | `feedback_engine.py` |
| Performance analytics | SQLite aggregation: form accuracy, rep accuracy, ROM%, consistency | `performance_logger.py` |
| Recommendation | Interpretable rule-driven weakness detection + difficulty progression | `recommender.py` |

### Exercise classifier training data

`train_classifier.py` ships with a **synthetic bootstrap**: it generates
labeled feature vectors from the same ideal angle ranges in
`exercise_rules.py` plus Gaussian noise, spanning both the top and bottom of
each rep. This gives a working classifier with **zero manual data
collection**. For production accuracy, capture real sessions with
`collect_training_data.py` and retrain with `--data-dir`:

```bash
python collect_training_data.py --exercise squat --seconds 30 --out data/real
python collect_training_data.py --exercise pushup --seconds 30 --out data/real
# ... repeat for lunge, plank, bicep_curl ...
python train_classifier.py --data-dir data/real
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# (Optional but recommended) train the classifier — otherwise a rule-based
# fallback classifier is used automatically for exercise auto-detection.
python train_classifier.py
```

## Running

### Browser UI (primary — see above for details)

```bash
uvicorn api:app --reload --port 8000
```
Open http://localhost:8000.

### Real-time webcam mode via OpenCV window (alternative)

```bash
# Fixed exercise (skips ML classification, most accurate rep counting):
python -m app.main --exercise squat --user_id bhumika

# Auto-detect exercise via the ML classifier:
python -m app.main --exercise auto --user_id bhumika

# Isometric hold (plank), with a 60s target:
python -m app.main --exercise plank --user_id bhumika --target_seconds 60
```

Controls: `q` to quit and save the session, `r` to reset the rep counter.

### Inference-only API without the browser UI (for your own frontend)

```bash
uvicorn api:app --reload --port 8000
```

```bash
curl -X POST localhost:8000/session/start \
  -H "Content-Type: application/json" \
  -d '{"user_id": "bhumika", "exercise": "squat"}'
# → {"session_id": "..."}

curl -X POST localhost:8000/session/<session_id>/analyze_frame \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<base64 jpeg>"}'
# → { "rep_count": 3, "form_score": 87.4, "feedback": ["Lower your hips..."], ... }

curl -X POST localhost:8000/session/<session_id>/end
curl "localhost:8000/users/bhumika/recommendation?exercise=squat"
```

Interactive API docs at `http://localhost:8000/docs` once running.

## Supported exercises

| Exercise | Rep-based? | Key checks |
|---|---|---|
| Squat | Yes | Depth (knee angle), back angle, knee valgus |
| Push-up | Yes | Elbow extension, body-line sag/pike |
| Lunge | Yes | Front-knee depth/tracking, torso lean |
| Plank | No (hold timer) | Body-line straightness (hip sag/pike) |
| Bicep curl | Yes | Elbow ROM, shoulder swing (strict-curl check) |

## Performance metrics computed per session

- **Form accuracy** — average posture confidence score across the session
- **Repetition accuracy** — % of reps that reached the target range of motion
- **Range of motion (ROM)** — average depth achieved vs. ideal target
- **Consistency** — inverse of rep-duration variance (steady tempo)
- **Completion %** — reps or hold-time vs. session target

## Project layout

```
gym_coach/
├── app/
│   ├── pose_estimator.py        # MediaPipe wrapper + angle geometry
│   ├── feature_engineering.py   # joint angles + ML feature vectors
│   ├── exercise_rules.py        # biomechanical ranges + FormChecker
│   ├── rep_counter.py           # state-machine rep counting, plank timer
│   ├── exercise_classifier.py   # RandomForest + rule-based fallback
│   ├── feedback_engine.py       # throttled coaching messages, EMA scoring
│   ├── performance_logger.py    # SQLite persistence
│   ├── recommender.py           # workout recommendation logic
│   └── main.py                  # real-time webcam CLI application
├── api.py                       # FastAPI: browser UI + inference-only endpoints
├── static/                      # Browser frontend (plain HTML/CSS/JS)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── train_classifier.py          # synthetic + real data training pipeline
├── collect_training_data.py     # capture real labeled data via webcam
├── docs/architecture.mmd        # Mermaid architecture diagram
├── models/                      # trained classifier artifacts (.joblib)
├── data/                        # SQLite DB + collected training data
└── requirements.txt
```

## Known limitations / next steps

- Single-person tracking only (MediaPipe Pose assumes one subject in frame).
- Camera angle affects angle accuracy — side-on view is best for
  squats/lunges/push-ups; a ~45° front-side view works for bicep curls.
- The synthetic classifier bootstrap is a cold-start convenience; accuracy
  on real footage improves substantially after `collect_training_data.py` +
  retraining.
- CLIP-based visual embeddings were considered for exercise recognition but
  intentionally deferred — the engineered angle-vector + RandomForest gives
  equivalent accuracy for this fixed exercise set at a fraction of the
  compute, making it viable for real-time CPU inference.
time CPU inference.

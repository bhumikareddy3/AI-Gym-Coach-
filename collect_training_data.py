"""
collect_training_data.py
---------------------------
Optional utility to capture REAL labeled feature vectors from your own
webcam to retrain the exercise classifier beyond the synthetic bootstrap
(see train_classifier.py). Perform the given exercise in front of the
camera; every detected frame is saved as a labeled .npy feature vector.

Usage:
    python collect_training_data.py --exercise squat --seconds 30 --out data/real
    python collect_training_data.py --exercise pushup --seconds 30 --out data/real
    ... repeat for lunge, plank, bicep_curl ...

Then retrain with:
    python train_classifier.py --data-dir data/real
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from app.pose_estimator import PoseEstimator
from app.feature_engineering import extract_feature_vector, is_fully_visible
from app.exercise_rules import SUPPORTED_EXERCISES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exercise", required=True, choices=SUPPORTED_EXERCISES)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--out", type=str, default="data/real")
    parser.add_argument("--camera_index", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out) / args.exercise
    out_dir.mkdir(parents=True, exist_ok=True)

    estimator = PoseEstimator()
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print(f"Recording '{args.exercise}' for {args.seconds}s. Perform the exercise now...")
    start = time.time()
    saved = 0

    try:
        while time.time() - start < args.seconds:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            lf = estimator.process(frame)
            if lf is not None and is_fully_visible(lf):
                vec = extract_feature_vector(lf)
                np.save(out_dir / f"{int(time.time() * 1000)}.npy", vec)
                saved += 1
                annotated = estimator.draw(frame, lf)
            else:
                annotated = frame
            remaining = max(0, args.seconds - (time.time() - start))
            cv2.putText(annotated, f"{args.exercise} | {remaining:0.1f}s left | saved {saved}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Collecting training data", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        estimator.close()

    print(f"Saved {saved} labeled samples to {out_dir}")


if __name__ == "__main__":
    main()

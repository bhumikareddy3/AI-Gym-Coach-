"""
train_classifier.py
---------------------
Trains the RandomForest exercise classifier used by app/exercise_classifier.py.

Two data sources are supported:

1. BOOTSTRAP (default, no data collection needed): generates a synthetic
   feature dataset from the same ideal joint-angle ranges defined in
   app/exercise_rules.py, with realistic Gaussian noise. This gives a
   working cold-start model with zero manual labeling, at the cost of not
   capturing every real-world visual quirk (camera angle, clothing, etc.)

2. REAL DATA (recommended before production use): point --data-dir at a
   folder of .npy feature vectors collected via collect_training_data.py
   (run while a labeled webcam session is active). Real data will always
   generalize better than the synthetic bootstrap.

Usage:
    python train_classifier.py                      # synthetic bootstrap only
    python train_classifier.py --data-dir data/real  # real + synthetic mixed
    python train_classifier.py --data-dir data/real --no-synthetic
"""

import argparse
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from app.exercise_rules import IDEAL_RANGES, SUPPORTED_EXERCISES
from app.exercise_classifier import DEFAULT_MODEL_PATH

RNG = np.random.default_rng(42)

# Feature order must match app/feature_engineering.FEATURE_NAMES exactly:
# elbow_L, elbow_R, shoulder_L, shoulder_R, hip_L, hip_R, knee_L, knee_R,
# torso_incline, hip_width_ratio, hand_to_shoulder_L, hand_to_shoulder_R,
# ankle_width_ratio, wrist_height_rel_shoulder_L, wrist_height_rel_shoulder_R

# Each exercise defined as (mean_vector, std_vector) across two phases
# (top-of-rep / bottom-of-rep) so the synthetic set spans the full ROM.
SYNTHETIC_PROFILES = {
    "squat": {
        "top":    dict(mean=[170, 170, 20, 20, 170, 170, 172, 172, 5,  0.9, 0.3, 0.3, 0.9, 0.1, 0.1], std=6),
        "bottom": dict(mean=[165, 165, 25, 25, 90,  90,  85,  85,  25, 0.9, 0.3, 0.3, 0.95,0.1, 0.1], std=6),
    },
    "pushup": {
        "top":    dict(mean=[170, 170, 60, 60, 175, 175, 175, 175, 80, 0.6, 0.9, 0.9, 0.6, -0.1,-0.1], std=6),
        "bottom": dict(mean=[85,  85,  55, 55, 175, 175, 175, 175, 80, 0.6, 0.5, 0.5, 0.6, -0.1,-0.1], std=6),
    },
    "lunge": {
        "top":    dict(mean=[170, 170, 20, 20, 165, 165, 168, 168, 10, 0.5, 0.3, 0.3, 1.6, 0.1, 0.1], std=6),
        "bottom": dict(mean=[168, 168, 20, 20, 140, 140, 90,  110, 15, 0.5, 0.3, 0.3, 1.6, 0.1, 0.1], std=6),
    },
    "plank": {
        "top":    dict(mean=[90, 90, 75, 75, 175, 175, 175, 175, 82, 0.6, 0.6, 0.6, 0.6, -0.1,-0.1], std=5),
        "bottom": dict(mean=[90, 90, 75, 75, 172, 172, 172, 172, 82, 0.6, 0.6, 0.6, 0.6, -0.1,-0.1], std=5),
    },
    "bicep_curl": {
        "top":    dict(mean=[45, 45, 15, 15, 175, 175, 172, 172, 3,  0.9, 0.55,0.55,0.9,  0.55,0.55], std=6),
        "bottom": dict(mean=[170,170, 15, 15, 175, 175, 172, 172, 3,  0.9, 0.85,0.85,0.9, -0.05,-0.05], std=6),
    },
}


def generate_synthetic_dataset(n_per_phase: int = 400):
    X, y = [], []
    for exercise, phases in SYNTHETIC_PROFILES.items():
        for phase_name, spec in phases.items():
            mean = np.array(spec["mean"], dtype=np.float64)
            std = spec["std"]
            samples = RNG.normal(loc=mean, scale=std, size=(n_per_phase, len(mean)))
            X.append(samples)
            y.extend([exercise] * n_per_phase)
    return np.vstack(X).astype(np.float32), np.array(y)


def load_real_dataset(data_dir: Path):
    """Expects data_dir/<exercise_name>/*.npy, each a (15,) feature vector
    saved by collect_training_data.py."""
    X, y = [], []
    for exercise_dir in sorted(data_dir.iterdir()):
        if not exercise_dir.is_dir():
            continue
        label = exercise_dir.name
        for npy_file in exercise_dir.glob("*.npy"):
            X.append(np.load(npy_file))
            y.append(label)
    if not X:
        return np.empty((0, 15), dtype=np.float32), np.empty((0,))
    return np.vstack(X).astype(np.float32), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None, help="Directory of real labeled .npy feature vectors")
    parser.add_argument("--no-synthetic", action="store_true", help="Exclude the synthetic bootstrap set")
    parser.add_argument("--out", type=str, default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    X_parts, y_parts = [], []

    if not args.no_synthetic:
        X_syn, y_syn = generate_synthetic_dataset()
        X_parts.append(X_syn)
        y_parts.append(y_syn)
        print(f"Synthetic samples: {len(y_syn)}")

    if args.data_dir:
        X_real, y_real = load_real_dataset(Path(args.data_dir))
        if len(y_real) > 0:
            X_parts.append(X_real)
            y_parts.append(y_real)
        print(f"Real samples: {len(y_real)}")

    if not X_parts:
        raise SystemExit("No training data available (use synthetic or --data-dir).")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    classes = clf.classes_
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "classes": classes}, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()

"""
performance_logger.py
------------------------
Persists workout telemetry (sessions, individual reps, posture scores) to a
local SQLite database so performance can be analyzed longitudinally and fed
into the WorkoutRecommender.

Schema:
  sessions(id, user_id, exercise, started_at, ended_at,
           form_accuracy, rep_accuracy, rom_pct, consistency, completion_pct)
  reps(id, session_id, rep_number, min_angle, max_angle, duration_sec, depth_ok)
"""

import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "gym_coach.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    exercise TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    form_accuracy REAL,
    rep_accuracy REAL,
    rom_pct REAL,
    consistency REAL,
    completion_pct REAL,
    total_reps INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS reps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    rep_number INTEGER NOT NULL,
    min_angle REAL,
    max_angle REAL,
    duration_sec REAL,
    depth_ok INTEGER,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);
"""


@dataclass
class SessionSummary:
    session_id: int
    user_id: str
    exercise: str
    form_accuracy: float
    rep_accuracy: float
    rom_pct: float
    consistency: float
    completion_pct: float
    total_reps: int


class PerformanceLogger:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def start_session(self, user_id: str, exercise: str) -> int:
        with closing(self._conn()) as conn:
            cur = conn.execute(
                "INSERT INTO sessions (user_id, exercise, started_at) VALUES (?, ?, ?)",
                (user_id, exercise, time.time()),
            )
            conn.commit()
            return cur.lastrowid

    def log_rep(self, session_id: int, rep_number: int, min_angle: float,
                max_angle: float, duration_sec: float, depth_ok: bool):
        with closing(self._conn()) as conn:
            conn.execute(
                "INSERT INTO reps (session_id, rep_number, min_angle, max_angle, duration_sec, depth_ok) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, rep_number, min_angle, max_angle, duration_sec, int(depth_ok)),
            )
            conn.commit()

    def end_session(self, session_id: int, form_accuracy: float, rep_accuracy: float,
                     rom_pct: float, consistency: float, completion_pct: float, total_reps: int):
        with closing(self._conn()) as conn:
            conn.execute(
                """UPDATE sessions SET ended_at=?, form_accuracy=?, rep_accuracy=?,
                   rom_pct=?, consistency=?, completion_pct=?, total_reps=? WHERE id=?""",
                (time.time(), form_accuracy, rep_accuracy, rom_pct, consistency,
                 completion_pct, total_reps, session_id),
            )
            conn.commit()

    def get_session_summary(self, session_id: int) -> Optional[SessionSummary]:
        with closing(self._conn()) as conn:
            row = conn.execute(
                "SELECT id, user_id, exercise, form_accuracy, rep_accuracy, rom_pct, "
                "consistency, completion_pct, total_reps FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return SessionSummary(*row)

    def get_user_history(self, user_id: str, limit: int = 20) -> List[SessionSummary]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT id, user_id, exercise, form_accuracy, rep_accuracy, rom_pct, "
                "consistency, completion_pct, total_reps FROM sessions "
                "WHERE user_id=? AND ended_at IS NOT NULL ORDER BY started_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [SessionSummary(*row) for row in rows]

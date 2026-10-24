"""SQLite prediction store written by the edge module and read by the dashboard."""
import sqlite3

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id     TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    temperature   REAL NOT NULL,
    humidity      REAL NOT NULL,
    occupancy     INTEGER NOT NULL,
    prev_kwh      REAL NOT NULL,
    predicted_kwh REAL NOT NULL,
    load_level    TEXT NOT NULL,
    inference_ms  REAL NOT NULL,
    predictor     TEXT NOT NULL,
    source        TEXT NOT NULL,
    received_at   TEXT NOT NULL
)
"""

COLUMNS = [
    "device_id", "timestamp", "temperature", "humidity", "occupancy", "prev_kwh",
    "predicted_kwh", "load_level", "inference_ms", "predictor", "source", "received_at",
]


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")  # lets the dashboard read while the edge writes
    conn.execute(SCHEMA)
    return conn


def insert_prediction(conn, record):
    placeholders = ", ".join("?" for _ in COLUMNS)
    conn.execute(
        f"INSERT INTO predictions ({', '.join(COLUMNS)}) VALUES ({placeholders})",
        [record[c] for c in COLUMNS],
    )
    conn.commit()


def recent_predictions(conn, limit=50):
    rows = conn.execute(
        f"SELECT {', '.join(COLUMNS)} FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(zip(COLUMNS, row)) for row in rows]


def clear_predictions(conn):
    conn.execute("DELETE FROM predictions")
    conn.commit()

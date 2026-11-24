import sqlite3
from contextlib import contextmanager
from backend.app import config

@contextmanager
def get_db():
    conn = sqlite3.connect(config.DATABASE_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, prediction INTEGER NOT NULL, probability REAL NOT NULL, model_name TEXT NOT NULL, model_version TEXT NOT NULL, created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))")

def save_prediction(result):
    with get_db() as conn:
        conn.execute("INSERT INTO predictions(prediction, probability, model_name, model_version) VALUES (?, ?, ?, ?)", tuple(result[k] for k in ("prediction", "probability", "model_name", "model_version")))

def get_predictions(limit):
    with get_db() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))]

def clear_predictions():
    with get_db() as conn:
        conn.execute("DELETE FROM predictions")

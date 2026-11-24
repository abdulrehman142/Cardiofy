"""Read the existing local MLflow SQLite store without SDK migrations or writes."""
import sqlite3
from contextlib import contextmanager
from fastapi import APIRouter, HTTPException, Query
from backend.app.config import ROOT
router = APIRouter(tags=["Experiments"])

@contextmanager
def store():
    path = ROOT / "mlflow.db"
    if not path.is_file():
        raise HTTPException(503, "Local MLflow database unavailable")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    except sqlite3.DatabaseError as exc:
        raise HTTPException(503, "MLflow database schema unavailable") from exc
    finally:
        conn.close()

@router.get("/experiments")
def experiments():
    with store() as conn:
        return [dict(row) for row in conn.execute("SELECT experiment_id AS id, name, lifecycle_stage FROM experiments WHERE lifecycle_stage='active' ORDER BY experiment_id")]

@router.get("/experiments/{experiment_id}/runs")
def runs(experiment_id: int, limit: int = Query(100, ge=1, le=500)):
    with store() as conn:
        if not conn.execute("SELECT 1 FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone():
            raise HTTPException(404, "Experiment not found")
        rows = conn.execute("SELECT run_uuid AS run_id, name AS run_name, status, start_time, end_time FROM runs WHERE experiment_id=? AND lifecycle_stage='active' ORDER BY start_time DESC LIMIT ?", (experiment_id, limit)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metrics"] = {r[0]:r[1] if not r[2] else None for r in conn.execute("SELECT key,value,is_nan FROM latest_metrics WHERE run_uuid=?", (row["run_id"],))}
            item["parameters"] = dict(conn.execute("SELECT key,value FROM params WHERE run_uuid=?", (row["run_id"],)))
            result.append(item)
        return result

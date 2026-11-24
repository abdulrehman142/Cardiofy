import json
import pandas as pd
from fastapi import HTTPException
from backend.app.config import REPORT_DIR, MODEL_DIR

def read_report(path):
    file = REPORT_DIR / path
    if not file.is_file():
        raise HTTPException(404, f"Report unavailable: {path}")
    df = pd.read_csv(file)
    df = df.rename(columns={c: "label" for c in df.columns if c.startswith("Unnamed:")})
    return json.loads(df.to_json(orient="records"))

def metadata():
    return json.loads((MODEL_DIR / "model_metadata.json").read_text())

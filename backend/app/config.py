from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models/final"
REPORT_DIR = ROOT / "reports"
DISCLAIMER = "Experimental machine learning prediction. Not a medical diagnosis."
DATABASE_PATH = Path(
    os.getenv(
        "DATABASE_PATH",
        str(ROOT / "backend/cardiofy.db")
    )
)

MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{ROOT / 'mlflow.db'}"
) 
DATA_PATH = ROOT / "data" / "processed" / "heart_disease_cleaned.csv"

MODEL_LIMITATION = (
    "This model is trained on a retrospective heart disease "
    "dataset and is intended for educational and experimental "
    "purposes only. It is not a medical diagnostic tool."
)
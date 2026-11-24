import json
from functools import lru_cache
import joblib
import pandas as pd
from backend.app.config import MODEL_DIR, DISCLAIMER, MODEL_LIMITATION

class ModelService:
    def __init__(self):
        self.metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text())
        self.model = joblib.load(MODEL_DIR / "cardiofy_model.joblib")

    def predict(self, patient):
        frame = pd.DataFrame([patient], columns=self.metadata["features"]).astype("float64")
        probability = float(self.model.predict_proba(frame)[0, list(self.model.classes_).index(1)])
        return dict(prediction=int(probability >= self.metadata["threshold"]), probability=probability,
                    threshold=self.metadata["threshold"], model_name=self.metadata["model_name"],
                    model_version=self.metadata["model_version"], disclaimer=DISCLAIMER, limitation=MODEL_LIMITATION)

@lru_cache(maxsize=1)
def get_model_service():
    return ModelService()

@lru_cache(maxsize=1)
def get_explainer():
    from ml.explainability import CardiofyExplainer
    return CardiofyExplainer()

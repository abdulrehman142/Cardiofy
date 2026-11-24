import joblib
import json
import pandas as pd

from preprocessing import ROOT

MODEL_DIR = ROOT / "models/final"

model = joblib.load(
    MODEL_DIR / "cardiofy_model.joblib"
)

with open(MODEL_DIR / "model_metadata.json") as file:
    metadata = json.load(file)

# Synthetic patient for software testing
patient = {
    "age": 55,
    "sex": 1,
    "cp": 3,
    "trestbps": 140,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 0,
    "thal": 3
}

X = pd.DataFrame(
    [patient],
    columns=metadata["features"]
).astype("float64")

probability = float(
    model.predict_proba(X)[0, 1]
)

prediction = int(
    probability >= metadata["threshold"]
)

print("Model:", metadata["model_name"])
print("Prediction:", prediction)
print(f"Positive probability: {probability:.2%}")
print(metadata["disclaimer"])
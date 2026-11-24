
import json
import joblib
import shap
import numpy as np
import pandas as pd

from preprocessing import ROOT, get_data_split

MODEL_DIR = ROOT / "models" / "final"

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Maximum Heart Rate",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Segment Slope",
    "ca": "Major Vessels",
    "thal": "Thallium Test"
}


class CardiofyExplainer:

    def __init__(self):

        self.model = joblib.load(
            MODEL_DIR / "cardiofy_model.joblib"
        )

        with open(
            MODEL_DIR / "model_metadata.json"
        ) as file:
            self.metadata = json.load(file)

        self.features = self.metadata["features"]

        X_train, _, _, _ = get_data_split()

        X_train = X_train[self.features].apply(
            pd.to_numeric, errors="coerce"
        ).astype("float64")

        # Background data from training partition only
        self.background = shap.sample(
            X_train,
            min(50, len(X_train)),
            random_state=42
        )

        self.explainer = shap.Explainer(
            self.predict_probability,
            self.background,
            algorithm="permutation",
            feature_names=self.features,
            seed=42
        )

    def predict_probability(self, data):

        X = pd.DataFrame(
            data,
            columns=self.features
        ).astype("float64")

        return self.model.predict_proba(X)[:, 1]

    def explain(self, data):

        X = pd.DataFrame(
            data,
            columns=self.features
        ).apply(
            pd.to_numeric,
            errors="coerce"
        ).astype("float64")

        return self.explainer(
            X,
            max_evals=10 * (2 * len(self.features) + 1)
        )

    def explain_patient(self, patient):

        X = pd.DataFrame(
            [patient],
            columns=self.features
        ).astype("float64")

        explanation = self.explain(X)[0]

        probability = float(
            self.predict_probability(X)[0]
        )

        contributions = []

        for index, feature in enumerate(self.features):

            value = X.iloc[0][feature]
            impact = float(explanation.values[index])

            contributions.append({
                "feature": feature,
                "label": FEATURE_LABELS.get(
                    feature, feature
                ),
                "value": (
                    None if pd.isna(value)
                    else float(value)
                ),
                "shap_value": impact,
                "impact_percentage_points": impact * 100,
                "direction": (
                    "increase" if impact > 0
                    else "decrease" if impact < 0
                    else "neutral"
                )
            })

        contributions.sort(
            key=lambda item: abs(item["shap_value"]),
            reverse=True
        )

        base_value = float(explanation.base_values)

        return {
            "model": self.metadata["model_name"],
            "probability": probability,
            "prediction": int(
                probability >= self.metadata["threshold"]
            ),
            "base_probability": base_value,
            "additivity_error": abs(
                probability
                - base_value
                - sum(explanation.values)
            ),
            "contributions": contributions,
            "disclaimer": self.metadata["disclaimer"]
        }


import json
import shap
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import ROOT, get_data_split
from explainability import CardiofyExplainer, FEATURE_LABELS

REPORT_DIR = ROOT / "reports" / "shap"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

engine = CardiofyExplainer()

X_train, _, _, _ = get_data_split()

X_sample = X_train.sample(
    n=min(100, len(X_train)),
    random_state=42
)

print("Generating SHAP explanations...")

shap_values = engine.explain(X_sample)

# Global feature importance
importance = pd.DataFrame({
    "feature": engine.features,
    "mean_abs_shap": np.abs(
        shap_values.values
    ).mean(axis=0)
})

importance["label"] = importance["feature"].map(
    FEATURE_LABELS
)

importance = importance.sort_values(
    "mean_abs_shap",
    ascending=False
)

importance.to_csv(
    REPORT_DIR / "feature_importance.csv",
    index=False
)

print("\nGLOBAL FEATURE IMPORTANCE\n")
print(importance.to_string(index=False))

# SHAP global bar plot
shap.plots.bar(
    shap_values,
    max_display=13,
    show=False
)

plt.tight_layout()
plt.savefig(
    REPORT_DIR / "shap_feature_importance.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close("all")

# SHAP beeswarm plot
shap.plots.beeswarm(
    shap_values,
    max_display=13,
    show=False
)

plt.tight_layout()
plt.savefig(
    REPORT_DIR / "shap_beeswarm.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close("all")

# Individual waterfall plot
shap.plots.waterfall(
    shap_values[0],
    max_display=13,
    show=False
)

plt.tight_layout()
plt.savefig(
    REPORT_DIR / "shap_waterfall.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close("all")

# SHAP heatmap
shap.plots.heatmap(
    shap_values,
    max_display=13,
    show=False
)

plt.tight_layout()
plt.savefig(
    REPORT_DIR / "shap_heatmap.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close("all")

# Synthetic patient for demonstration
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

explanation = engine.explain_patient(patient)

with open(
    REPORT_DIR / "sample_explanation.json", "w"
) as file:
    json.dump(explanation, file, indent=4)

# Track explainability artifacts in MLflow
mlflow.set_tracking_uri(
    f"sqlite:///{ROOT / 'mlflow.db'}"
)

mlflow.set_experiment(
    "Cardiofy - SHAP Explainability"
)

with mlflow.start_run(
    run_name="Final Model Explainability"
):

    mlflow.log_params({
        "model": engine.metadata["model_name"],
        "explainer": "Permutation SHAP",
        "background_samples": len(engine.background),
        "explained_samples": len(X_sample)
    })

    for _, row in importance.iterrows():

        mlflow.log_metric(
            f"importance_{row['feature']}",
            float(row["mean_abs_shap"])
        )

    mlflow.log_artifacts(
        str(REPORT_DIR),
        artifact_path="shap"
    )

print("\nSHAP reports generated successfully.")


import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.calibration import (
    CalibratedClassifierCV,
    CalibrationDisplay
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from preprocessing import (
    ROOT,
    FEATURES,
    get_data_split,
    build_preprocessor
)

from model_registry import get_models
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration",
    "sklearn.model_selection._split.StratifiedKFold"
]
# Directories
REPORT_DIR = ROOT / "reports" / "evaluation"
MODEL_DIR = ROOT / "models" / "final"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# MLflow configuration
mlflow.set_tracking_uri(
    f"sqlite:///{ROOT / 'mlflow.db'}"
)

mlflow.set_experiment(
    "Cardiofy - Final Evaluation"
)

# Load training and testing data
X_train, X_test, y_train, y_test = get_data_split()

X_train = X_train.apply(
    pd.to_numeric, errors="coerce"
).astype("float64")

X_test = X_test.apply(
    pd.to_numeric, errors="coerce"
).astype("float64")

# Load previous results
baseline = pd.read_csv(
    ROOT / "reports/training/baseline_results.csv"
)

tuned = pd.read_csv(
    ROOT / "reports/tuning/tuned_results.csv"
)

baseline["stage"] = "baseline"
tuned["stage"] = "tuned"

candidates = pd.concat(
    [baseline, tuned],
    ignore_index=True
)

# Exclude reference classifier
candidates = candidates[
    candidates["model"] != "Dummy Classifier"
]

candidates = candidates.sort_values(
    ["roc_auc", "recall", "f1"],
    ascending=False
)

winner = candidates.iloc[0]

best_name = winner["model"]
best_stage = winner["stage"]

filename = best_name.lower().replace(" ", "_")

print("Selected model:", best_name)
print("Selected stage:", best_stage)
print("CV ROC-AUC:", winner["roc_auc"])

# Reconstruct selected pipeline
pipeline = Pipeline([
    ("preprocessor", build_preprocessor()),
    ("classifier", get_models()[best_name])
])

best_params = {}

if best_stage == "tuned":

    params_path = (
        ROOT / "reports/tuning"
        / f"{filename}_params.json"
    )

    with open(params_path) as file:
        best_params = json.load(file)

    pipeline.set_params(**best_params)

# Calibration configuration
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

calibrated_model = CalibratedClassifierCV(
    estimator=pipeline,
    method="sigmoid",
    cv=cv
)

# Train using training data only
pipeline.fit(X_train, y_train)
calibrated_model.fit(X_train, y_train)

# Evaluation function
def evaluate(model):

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    metrics = {
        "accuracy": accuracy_score(
            y_test, predictions
        ),
        "precision": precision_score(
            y_test, predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test, probabilities
        ),
        "pr_auc": average_precision_score(
            y_test, probabilities
        ),
        "brier_loss": brier_score_loss(
            y_test, probabilities
        ),
        "specificity": (
            tn / (tn + fp)
            if tn + fp else 0
        )
    }

    return metrics, predictions, probabilities

# Evaluate original and calibrated models
original_metrics, original_pred, original_prob = (
    evaluate(pipeline)
)

final_metrics, final_pred, final_prob = (
    evaluate(calibrated_model)
)

# Save evaluation comparison
comparison = pd.DataFrame([
    {
        "version": "Uncalibrated",
        **original_metrics
    },
    {
        "version": "Calibrated",
        **final_metrics
    }
])

comparison.to_csv(
    REPORT_DIR / "calibration_comparison.csv",
    index=False
)

print("\n========== FINAL RESULTS ==========")

print(comparison.round(4).to_string(index=False))

# Predetermined final model
final_model = calibrated_model

# Save final pipeline
joblib.dump(
    final_model,
    MODEL_DIR / "cardiofy_model.joblib"
)

# Save original pipeline for analysis
joblib.dump(
    pipeline,
    MODEL_DIR / "uncalibrated_model.joblib"
)

# Model metadata
metadata = {
    "model_name": best_name,
    "stage": best_stage,
    "model_version": "1.0.0",
    "calibration": "sigmoid",
    "threshold": 0.5,
    "features": FEATURES,
    "hyperparameters": best_params,
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "selection_cv_roc_auc": float(
        winner["roc_auc"]
    ),
    "metrics": final_metrics,
    "disclaimer": (
        "Experimental prediction system. "
        "Not a medical diagnosis."
    )
}

with open(
    MODEL_DIR / "model_metadata.json", "w"
) as file:
    json.dump(metadata, file, indent=4)

# Save classification report
report = classification_report(
    y_test,
    final_pred,
    labels=[0, 1],
    target_names=["No Disease", "Disease"],
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).T.to_csv(
    REPORT_DIR / "classification_report.csv"
)

# Save evaluation predictions
pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": final_pred,
    "probability": final_prob
}).to_csv(
    REPORT_DIR / "test_predictions.csv",
    index=False
)

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    final_pred,
    labels=[0, 1],
    display_labels=["No Disease", "Disease"],
    cmap="Blues"
)

plt.title("Cardiofy - Confusion Matrix")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "confusion_matrix.png",
    dpi=150
)
plt.close()

# ROC curve
RocCurveDisplay.from_predictions(
    y_test,
    final_prob,
    name=best_name
)

plt.plot([0, 1], [0, 1], "--", color="gray")
plt.title("Cardiofy - ROC Curve")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "roc_curve.png",
    dpi=150
)
plt.close()

# Precision-recall curve
PrecisionRecallDisplay.from_predictions(
    y_test,
    final_prob,
    name=best_name
)

plt.title("Cardiofy - Precision Recall Curve")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "precision_recall_curve.png",
    dpi=150
)
plt.close()

# Calibration curve
fig, ax = plt.subplots(figsize=(8, 6))

CalibrationDisplay.from_predictions(
    y_test,
    original_prob,
    n_bins=5,
    name="Uncalibrated",
    ax=ax
)

CalibrationDisplay.from_predictions(
    y_test,
    final_prob,
    n_bins=5,
    name="Calibrated",
    ax=ax
)

plt.title("Cardiofy - Calibration Curve")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "calibration_curve.png",
    dpi=150
)
plt.close()

# Track final experiment
with mlflow.start_run(
    run_name=f"{best_name} - Final Model"
):

    mlflow.log_params({
        "model": best_name,
        "stage": best_stage,
        "calibration": "sigmoid",
        "threshold": 0.5,
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    })

    mlflow.log_metrics(final_metrics)

    mlflow.log_metrics({
        f"uncalibrated_{key}": value
        for key, value in original_metrics.items()
    })

    mlflow.set_tag("stage", "final_evaluation")

    mlflow.log_artifacts(
        str(REPORT_DIR),
        artifact_path="evaluation"
    )

    mlflow.log_artifact(
        str(MODEL_DIR / "model_metadata.json")
    )

    mlflow.sklearn.log_model(
        sk_model=final_model,
        name="model",
        skops_trusted_types=SKOPS_TRUSTED_TYPES
    )

print("\nFinal model saved successfully.")
print("Model:", best_name)
print("Location:", MODEL_DIR)

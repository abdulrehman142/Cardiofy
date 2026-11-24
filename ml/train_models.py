import time
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)

from preprocessing import (
    ROOT,
    get_data_split,
    build_preprocessor
)

from model_registry import get_models

REPORT_DIR = ROOT / "reports" / "training"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

X_train, X_test, y_train, y_test = get_data_split()

models = get_models()

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "brier_loss": "neg_brier_score"
}

results = []

for name, model in models.items():

    print(f"\nTraining: {name}")

    start = time.time()

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", model)
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise"
    )

    result = {
        "model": name,
        "training_time": round(time.time() - start, 2)
    }

    for metric in scoring:

        values = scores[f"test_{metric}"]

        if metric == "brier_loss":
            values = -values

        result[metric] = values.mean()
        result[f"{metric}_std"] = values.std()

    results.append(result)

    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Recall:   {result['recall']:.4f}")
    print(f"F1-score: {result['f1']:.4f}")
    print(f"ROC-AUC:  {result['roc_auc']:.4f}")

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="roc_auc",
    ascending=False
)

results_df.to_csv(
    REPORT_DIR / "baseline_results.csv",
    index=False
)

print("\n========== MODEL LEADERBOARD ==========\n")

print(results_df[
    [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_loss"
    ]
].to_string(index=False))

print("\nTraining completed successfully.")
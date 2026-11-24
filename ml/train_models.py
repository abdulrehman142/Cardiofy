
import time
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)
from sklearn.metrics import make_scorer, precision_score

from preprocessing import (
    ROOT,
    get_data_split,
    build_preprocessor
)

from model_registry import get_models

# Configuration
REPORT_DIR = ROOT / "reports" / "training"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(
    f"sqlite:///{ROOT / 'mlflow.db'}"
)

mlflow.set_experiment("Cardiofy - Baseline Models")

# Load dataset
X_train, X_test, y_train, y_test = get_data_split()

models = get_models()

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(
        precision_score, zero_division=0
    ),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "brier_loss": "neg_brier_score"
}

results = []

for name, model in models.items():

    print(f"\nTraining: {name}")

    with mlflow.start_run(run_name=name) as run:

        start = time.time()

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", model)
        ])

        # Track experiment configuration
        mlflow.log_param("model", name)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("input_features", X_train.shape[1])

        mlflow.log_params({
            f"classifier__{key}": str(value)
            for key, value in model.get_params().items()
        })

        mlflow.set_tag("stage", "baseline")
        mlflow.set_tag("project", "Cardiofy")

        # Cross-validation
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
            "run_id": run.info.run_id
        }

        for metric in scoring:

            values = scores[f"test_{metric}"]

            if metric == "brier_loss":
                values = -values

            result[metric] = float(values.mean())
            result[f"{metric}_std"] = float(values.std())

            mlflow.log_metric(
                metric, result[metric]
            )

            mlflow.log_metric(
                f"{metric}_std",
                result[f"{metric}_std"]
            )

            # Track individual fold results
            for fold, value in enumerate(values, start=1):
                mlflow.log_metric(
                    f"{metric}_fold",
                    float(value),
                    step=fold
                )

        # Record cross-validation runtime
        result["training_time"] = round(
            time.time() - start, 2
        )

        mlflow.log_metric(
            "training_time",
            result["training_time"]
        )

        # Fit baseline model on full training dataset
        pipeline.fit(X_train, y_train)

        # Save complete pipeline in MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=X_train.head(3),
            skops_trusted_types=[
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier"
]
)

        results.append(result)

        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Recall:   {result['recall']:.4f}")
        print(f"F1-score: {result['f1']:.4f}")
        print(f"ROC-AUC:  {result['roc_auc']:.4f}")

# Create leaderboard
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

print("\nAll experiments logged to MLflow.")

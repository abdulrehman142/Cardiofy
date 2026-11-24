
import time
import pandas as pd
import mlflow

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    LeaveOneGroupOut,
    RandomizedSearchCV,
    StratifiedKFold,
    ParameterGrid
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)

from preprocessing import (
    ROOT,
    FEATURES,
    load_data,
    build_preprocessor
)
from model_registry import get_models
from hyperparameters import PARAM_GRIDS

REPORT_DIR = ROOT / "reports" / "hospital_validation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(
    f"sqlite:///{ROOT / 'mlflow.db'}"
)

mlflow.set_experiment(
    "Cardiofy - Hospital Validation"
)

df = load_data()
X = df[FEATURES].copy()

numerical_columns = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak"
]

X[numerical_columns] = X[numerical_columns].astype("float64")
y = df["target"]
groups = df["dataset"]

logo = LeaveOneGroupOut()

inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for name, model in get_models().items():

    print(f"\nEvaluating: {name}")

    for train_idx, test_idx in logo.split(X, y, groups):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        hospital = groups.iloc[test_idx].iloc[0]

        print(f"Testing on: {hospital}")

        start = time.time()

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", model)
        ])

        with mlflow.start_run(
            run_name=f"{name} - {hospital}"
        ):

            if name in PARAM_GRIDS:

                grid = PARAM_GRIDS[name]

                search = RandomizedSearchCV(
                    pipeline,
                    param_distributions=grid,
                    n_iter=min(
                        20, len(ParameterGrid(grid))
                    ),
                    scoring="roc_auc",
                    cv=inner_cv,
                    random_state=42,
                    n_jobs=1,
                    error_score="raise"
                )

                search.fit(X_train, y_train)

                fitted_model = search.best_estimator_

                mlflow.log_params(search.best_params_)

            else:

                pipeline.fit(X_train, y_train)
                fitted_model = pipeline

            predictions = fitted_model.predict(X_test)

            probabilities = fitted_model.predict_proba(
                X_test
            )[:, 1]

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
                "brier_loss": brier_score_loss(
                    y_test, probabilities
                ),
                "specificity": (
                    tn / (tn + fp)
                    if tn + fp > 0 else 0
                ),
                "training_time": time.time() - start
            }

            if y_test.nunique() == 2:

                metrics["roc_auc"] = roc_auc_score(
                    y_test, probabilities
                )

                metrics["pr_auc"] = (
                    average_precision_score(
                        y_test, probabilities
                    )
                )

            mlflow.log_metrics(metrics)

            mlflow.log_params({
                "model": name,
                "test_hospital": hospital,
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            })

            mlflow.set_tag("stage", "hospital_validation")

            results.append({
                "model": name,
                "hospital": hospital,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                **metrics
            })

            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1: {metrics['f1']:.4f}")

results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "hospital_results.csv",
    index=False
)

metric_columns = [
    "accuracy", "precision", "recall",
    "f1", "roc_auc", "pr_auc",
    "brier_loss", "specificity"
]

summary = results_df.groupby("model")[
    metric_columns
].agg(["mean", "std"])

summary.columns = [
    f"{metric}_{stat}"
    for metric, stat in summary.columns
]

summary = summary.sort_values(
    "roc_auc_mean",
    ascending=False
)

summary.to_csv(
    REPORT_DIR / "hospital_summary.csv"
)

print("\n========== HOSPITAL LEADERBOARD ==========")
print(summary.round(4).to_string())

print("\nHospital validation completed.")

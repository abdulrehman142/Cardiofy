
import time
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    ParameterGrid
)
from sklearn.metrics import (
    make_scorer,
    precision_score
)

from preprocessing import (
    ROOT,
    get_data_split,
    build_preprocessor
)

from model_registry import get_models
from hyperparameters import PARAM_GRIDS
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier"
]
# Configuration
RANDOM_STATE = 42
N_ITER = 20

REPORT_DIR = ROOT / "reports" / "tuning"
MODEL_DIR = ROOT / "models" / "tuned"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(
    f"sqlite:///{ROOT / 'mlflow.db'}"
)

mlflow.set_experiment(
    "Cardiofy - Hyperparameter Tuning"
)

# Load dataset
X_train, X_test, y_train, y_test = get_data_split()

# Consistent numeric input types for MLflow
numerical_columns = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak"
]

X_train = X_train.copy()
X_test = X_test.copy()

X_train[numerical_columns] = (
    X_train[numerical_columns].astype("float64")
)

X_test[numerical_columns] = (
    X_test[numerical_columns].astype("float64")
)

models = get_models()

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(
        precision_score,
        zero_division=0
    ),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "brier_loss": "neg_brier_score"
}

results = []

for name, model in models.items():

    if name not in PARAM_GRIDS:
        continue

    print(f"\nOptimizing: {name}")

    start = time.time()

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", model)
    ])

    param_grid = PARAM_GRIDS[name]

    n_iter = min(
        N_ITER,
        len(ParameterGrid(param_grid))
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        refit="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
        error_score="raise",
        return_train_score=False
    )

    with mlflow.start_run(
        run_name=f"{name} - Tuned"
    ) as run:

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        best_index = search.best_index_

        result = {
            "model": name,
            "run_id": run.info.run_id,
            "training_time": round(
                time.time() - start, 2
            )
        }

        for metric in scoring:

            value = float(
                search.cv_results_[
                    f"mean_test_{metric}"
                ][best_index]
            )

            std = float(
                search.cv_results_[
                    f"std_test_{metric}"
                ][best_index]
            )

            if metric == "brier_loss":
                value = -value

            result[metric] = value
            result[f"{metric}_std"] = std

            mlflow.log_metric(metric, value)
            mlflow.log_metric(
                f"{metric}_std", std
            )

        mlflow.log_params(search.best_params_)
        mlflow.log_param("model", name)
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("random_state", RANDOM_STATE)

        mlflow.log_metric(
            "training_time",
            result["training_time"]
        )

        mlflow.set_tag("stage", "tuning")
        mlflow.set_tag("project", "Cardiofy")

        # Save tuned pipeline
        filename = name.lower().replace(" ", "_")

        model_path = MODEL_DIR / f"{filename}.joblib"

        joblib.dump(best_model, model_path)

        # Save best parameters
        params_path = REPORT_DIR / f"{filename}_params.json"

        with open(params_path, "w") as file:
            json.dump(
                search.best_params_,
                file,
                indent=4,
                default=str
            )

        # Save all search results
        search_path = REPORT_DIR / f"{filename}_search.csv"

        pd.DataFrame(
            search.cv_results_
        ).to_csv(search_path, index=False)

        mlflow.log_artifact(str(params_path))
        mlflow.log_artifact(str(search_path))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            input_example=X_train.head(3),
            skops_trusted_types=SKOPS_TRUSTED_TYPES
        )

        results.append(result)

        print("Best parameters:", search.best_params_)
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Recall:   {result['recall']:.4f}")
        print(f"F1-score: {result['f1']:.4f}")
        print(f"ROC-AUC:  {result['roc_auc']:.4f}")

# Save tuned leaderboard
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="roc_auc",
    ascending=False
)

results_df.to_csv(
    REPORT_DIR / "tuned_results.csv",
    index=False
)

print("\n========== TUNED MODEL LEADERBOARD ==========\n")

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

print("\nHyperparameter tuning completed.")

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/raw/heart_disease.csv"

RANDOM_STATE = 42

NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak"
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# DATA LOADING AND CLEANING
# ============================================================

def load_data():

    df = pd.read_csv(DATA_PATH, na_values="?")

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Support alternative column naming conventions
    df = df.rename(columns={
        "thalach": "thalch",
        "hospital": "dataset"
    })

    # Remove exact duplicates
    df = df.drop_duplicates().copy()

    # Validate required features
    missing_columns = [
        column for column in FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Convert numerical features
    for column in NUMERICAL_FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Preserve categorical features
    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].apply(
            lambda value: str(value).strip()
            if pd.notna(value)
            else np.nan
        ).astype(object)

    # Handle invalid medical measurements
    invalid_zero_columns = [
        "age",
        "trestbps",
        "chol",
        "thalch"
    ]

    for column in invalid_zero_columns:
        df.loc[df[column] <= 0, column] = np.nan

    # Replace infinite numerical values
    df[NUMERICAL_FEATURES] = df[
        NUMERICAL_FEATURES
    ].replace([np.inf, -np.inf], np.nan)

    # Convert original target into binary classification
    if "num" in df.columns:

        target = pd.to_numeric(
            df["num"],
            errors="coerce"
        )

        df = df.loc[
            target.isin([0, 1, 2, 3, 4])
        ].copy()

        df["target"] = (
            target.loc[df.index] > 0
        ).astype(int)

    elif "target" in df.columns:

        df["target"] = pd.to_numeric(
            df["target"],
            errors="coerce"
        )

        df = df[
            df["target"].isin([0, 1])
        ].copy()

        df["target"] = df["target"].astype(int)

    else:
        raise ValueError("Target column not found.")

    return df.reset_index(drop=True)


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

def build_preprocessor():

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    # Combine numerical and categorical pipelines
    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            NUMERICAL_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ], remainder="drop")

    return preprocessor


# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

def get_data_split():

    df = load_data()

    X = df[FEATURES]
    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )


# ============================================================
# PREPROCESSING VERIFICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CARDIOFY DATA PREPROCESSING")
    print("=" * 60)

    df = load_data()

    print("\nCleaned dataset shape:", df.shape)

    print("\nDataset columns:")
    print(df.columns.tolist())

    print("\nTarget distribution:")
    print(df["target"].value_counts())

    print("\nMissing values before preprocessing:")
    print(df[FEATURES].isna().sum())

    # Split dataset
    X_train, X_test, y_train, y_test = get_data_split()

    # Initialize preprocessor
    preprocessor = build_preprocessor()

    # Fit only on training data to prevent data leakage
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data using fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)

    print("\nOriginal shapes:")
    print("Training:", X_train.shape)
    print("Testing:", X_test.shape)

    print("\nProcessed shapes:")
    print("Training:", X_train_processed.shape)
    print("Testing:", X_test_processed.shape)

    print("\nMissing values after preprocessing:")
    print("Training:", np.isnan(X_train_processed).sum())
    print("Testing:", np.isnan(X_test_processed).sum())

    print("\nTraining class distribution (%):")
    print(
        (y_train.value_counts(normalize=True) * 100)
        .round(2)
    )

    print("\nTesting class distribution (%):")
    print(
        (y_test.value_counts(normalize=True) * 100)
        .round(2)
    )

    print("\nTransformed feature names:")

    for feature in preprocessor.get_feature_names_out():
        print(feature)

    # Verify preprocessing results
    assert not np.isnan(X_train_processed).any(), (
        "Training data contains missing values."
    )

    assert not np.isnan(X_test_processed).any(), (
        "Testing data contains missing values."
    )

    assert np.isfinite(X_train_processed).all(), (
        "Training data contains non-finite values."
    )

    assert np.isfinite(X_test_processed).all(), (
        "Testing data contains non-finite values."
    )

    print("\n" + "=" * 60)
    print("Preprocessing completed successfully.")
    print("=" * 60)
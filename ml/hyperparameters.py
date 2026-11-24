
PARAM_GRIDS = {

    "Logistic Regression": {
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__solver": ["lbfgs", "liblinear"],
        "classifier__class_weight": [None, "balanced"]
    },

    "KNN": {
        "classifier__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "classifier__weights": ["uniform", "distance"],
        "classifier__p": [1, 2]
    },

    "Decision Tree": {
        "classifier__max_depth": [3, 5, 7, 10, 15, None],
        "classifier__min_samples_split": [2, 5, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 4, 8],
        "classifier__criterion": ["gini", "entropy"]
    },

    "Random Forest": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [3, 5, 10, 15, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
        "classifier__class_weight": [None, "balanced"]
    },

    "SVM": {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__kernel": ["linear", "rbf"],
        "classifier__gamma": ["scale", "auto"],
        "classifier__class_weight": [None, "balanced"]
    },

    "Gradient Boosting": {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__max_depth": [2, 3, 4, 5],
        "classifier__subsample": [0.7, 0.8, 1.0]
    },

    "XGBoost": {
        "classifier__n_estimators": [50, 100, 200, 300],
        "classifier__max_depth": [2, 3, 4, 5, 6],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.7, 0.8, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 1.0],
        "classifier__reg_alpha": [0, 0.1, 1],
        "classifier__reg_lambda": [1, 5, 10]
    },

    "LightGBM": {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__num_leaves": [7, 15, 31],
        "classifier__max_depth": [3, 5, -1],
        "classifier__min_child_samples": [10, 20, 30],
        "classifier__reg_alpha": [0, 0.1, 1],
        "classifier__reg_lambda": [0, 1, 5]
    }
}

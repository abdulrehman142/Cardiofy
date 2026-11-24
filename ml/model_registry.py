from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42


def get_models():

    return {
        "Dummy Classifier": DummyClassifier(
            strategy="most_frequent"
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE
        ),

        "SVM": SVC(
            probability=True,
            random_state=RANDOM_STATE
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),

        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0
        ),

        "LightGBM": LGBMClassifier(
            random_state=RANDOM_STATE,
            verbosity=-1
        )
    }

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import ROOT

REPORT_DIR = ROOT / "reports" / "hospital_validation"

df = pd.read_csv(
    REPORT_DIR / "hospital_results.csv"
)

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier_loss",
    "specificity"
]

for metric in metrics:

    pivot = df.pivot(
        index="model",
        columns="hospital",
        values=metric
    )

    plt.figure(figsize=(12, 7))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap=(
            "RdYlGn_r"
            if metric == "brier_loss"
            else "RdYlGn"
        ),
        vmin=0,
        vmax=1
    )

    plt.title(
        f"Hospital-wise {metric.upper()}"
    )

    plt.xlabel("Unseen Hospital")
    plt.ylabel("ML Model")

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / f"{metric}_heatmap.png",
        dpi=150
    )

    plt.close()

print("Hospital validation graphs saved.")


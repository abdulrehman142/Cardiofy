
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import ROOT

REPORT_DIR = ROOT / "reports" / "comparison"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

baseline = pd.read_csv(
    ROOT / "reports/training/baseline_results.csv"
)

tuned = pd.read_csv(
    ROOT / "reports/tuning/tuned_results.csv"
)

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier_loss"
]

comparison = baseline[
    ["model"] + metrics
].merge(
    tuned[["model"] + metrics],
    on="model",
    suffixes=("_baseline", "_tuned")
)

for metric in metrics:

    if metric == "brier_loss":
        comparison[f"{metric}_improvement"] = (
            comparison[f"{metric}_baseline"]
            - comparison[f"{metric}_tuned"]
        )
    else:
        comparison[f"{metric}_improvement"] = (
            comparison[f"{metric}_tuned"]
            - comparison[f"{metric}_baseline"]
        )

comparison = comparison.sort_values(
    "roc_auc_tuned",
    ascending=False
)

comparison.to_csv(
    REPORT_DIR / "baseline_vs_tuned.csv",
    index=False
)

print(comparison[
    [
        "model",
        "roc_auc_baseline",
        "roc_auc_tuned",
        "roc_auc_improvement"
    ]
].to_string(index=False))

# Generate comparison charts
for metric in metrics:

    chart_data = comparison.set_index("model")[
        [
            f"{metric}_baseline",
            f"{metric}_tuned"
        ]
    ]

    chart_data.columns = ["Baseline", "Tuned"]

    ax = chart_data.plot(
        kind="bar",
        figsize=(12, 6),
        rot=30
    )

    ax.set_title(
        f"Baseline vs Tuned - {metric.upper()}"
    )

    ax.set_ylabel(metric)
    ax.set_xlabel("Model")
    ax.legend()

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / f"{metric}_comparison.png",
        dpi=150
    )

    plt.close()

print("\nComparison reports and graphs saved.")

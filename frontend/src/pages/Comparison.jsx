import React, { useState } from "react";
import { useData, State, pct, fmt } from "../components/ui";
const metrics = {
  roc_auc: "ROC-AUC",
  recall: "Recall",
  accuracy: "Accuracy",
  f1: "F1 score",
  brier_loss: "Brier score",
};
export default function Comparison() {
  const comparison = useData("/comparison");
  const models = useData("/models");
  const [metric, setMetric] = useState("roc_auc");
  const value = (number) =>
    metric === "brier_loss" ? fmt(number) : pct(number);
  return (
    <>
      <section className="intro comparison-intro">
        <h1>Model comparison</h1>
        <p>Compare baseline and tuned performance across models.</p>
      </section>
      <section
        className="comparison-section"
        aria-labelledby="comparison-title"
      >
        <div className="section-heading">
          <div>
            <h2 id="comparison-title">Model performance</h2>
            <p>Cross-validation on the training data.</p>
          </div>
          <label className="metric-select">
            Compare by
            <select
              value={metric}
              onChange={(event) => setMetric(event.target.value)}
            >
              {Object.entries(metrics).map(([key, label]) => (
                <option value={key} key={key}>
                  {label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <State state={comparison}>
          {(rows) => (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Baseline</th>
                    <th>Tuned</th>
                    <th>Change</th>
                  </tr>
                </thead>
                <tbody>
                  {[...rows]
                    .sort((a, b) =>
                      metric === "brier_loss"
                        ? a[`${metric}_tuned`] - b[`${metric}_tuned`]
                        : b[`${metric}_tuned`] - a[`${metric}_tuned`],
                    )
                    .map((row) => {
                      const baseline = row[`${metric}_baseline`],
                        tuned = row[`${metric}_tuned`];
                      const delta = tuned - baseline;
                      const better =
                        metric === "brier_loss" ? delta < 0 : delta > 0;
                      return (
                        <tr key={row.model}>
                          <td>
                            <span>{row.model}</span>
                            {models.data?.selected === row.model && (
                              <span className="selected-model">Selected</span>
                            )}
                          </td>
                          <td>{value(baseline)}</td>
                          <td>{value(tuned)}</td>
                          <td>
                            <span
                              className={
                                Math.abs(delta) < 1e-8
                                  ? "delta neutral"
                                  : better
                                    ? "delta better"
                                    : "delta worse"
                              }
                            >
                              {Math.abs(delta) < 1e-8
                                ? "No change"
                                : `${delta > 0 ? "+" : ""}${metric === "brier_loss" ? delta.toFixed(4) : `${(delta * 100).toFixed(2)} pp`}`}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          )}
        </State>
        <p className="table-note">
          {metric === "brier_loss"
            ? "Lower scores indicate better probability accuracy."
            : "Higher scores indicate better performance for this metric."}{" "}
          Changing this view does not change the saved prediction model.
        </p>
      </section>
      <details className="model-note">
        <summary>How to read these results</summary>
        <div>
          <p>
            ROC-AUC measures how well the model separates the dataset’s two
            classes. Recall measures how many positive cases it detects.
            Accuracy is the fraction of correct labels. F1 balances precision
            and recall. Brier score measures probability error.
          </p>
          <p>
            These are cross-validation results used during model development,
            not held-out test scores or evidence of clinical validation. The
            saved SVM was selected using training ROC-AUC, then
            sigmoid-calibrated.
          </p>
          <p>
            The current prediction artifact dropped seven categorical features
            during training. Its output depends on the six measurements shown on
            the prediction page. The underlying model and reports have been
            preserved.
          </p>
        </div>
      </details>
    </>
  );
}

import React, { useEffect, useState } from "react";
import { HeartPulse, ChevronRight, CircleAlert, RefreshCw } from "lucide-react";
import { api } from "../api";
import {
  pct,
  fmt,
  useData,
  State,
  Panel,
  Table,
  Bars,
  Stats,
} from "../components/ui";
export default function Evaluation() {
  const state = useData("/metrics");
  return (
    <State state={state}>
      {(d) => (
        <>
          <Stats
            items={Object.entries(d.metadata.metrics).map(([k, v]) => [
              k.replaceAll("_", " "),
              k === "brier_loss" ? v.toFixed(4) : pct(v),
              "Held-out test partition",
            ])}
          />
          <Panel
            title="Calibration comparison"
            subtitle="Saved uncalibrated and sigmoid-calibrated test results"
          >
            <Table
              rows={d.calibration}
              columns={["version", "roc_auc", "recall", "f1", "brier_loss"]}
            />
          </Panel>
          <Panel
            title="Classification report"
            subtitle={`${d.metadata.test_samples} held-out test records`}
          >
            <Table
              rows={d.classification}
              columns={["label", "precision", "recall", "f1-score", "support"]}
            />
          </Panel>
          <div className="grid two">
            {[
              "confusion_matrix",
              "roc_curve",
              "precision_recall_curve",
              "calibration_curve",
            ].map((name) => (
              <Panel title={name.replaceAll("_", " ")} key={name}>
                <img
                  className="plot"
                  src={`/api/plots/evaluation/${name}.png`}
                  alt={name.replaceAll("_", " ")}
                  onError={(e) => {
                    e.currentTarget.alt = "Plot unavailable";
                  }}
                />
              </Panel>
            ))}
          </div>
          <div className="notice">{d.limitation}</div>
        </>
      )}
    </State>
  );
}

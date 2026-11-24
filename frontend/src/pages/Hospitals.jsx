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
export default function Hospitals() {
  const state = useData("/hospitals"),
    [hospital, setHospital] = useState("All");
  return (
    <State state={state}>
      {(d) => (
        <>
          <div className="notice">
            Leave-one-hospital-out validation is a separate retrospective
            generalization experiment. These results do not establish clinical
            effectiveness or select the final model.
          </div>
          <Panel
            title="Generalization across hospitals"
            subtitle="Mean and standard deviation across held-out hospital folds"
          >
            <Table
              rows={d.summary}
              columns={[
                "model",
                "roc_auc_mean",
                "roc_auc_std",
                "recall_mean",
                "specificity_mean",
                "brier_loss_mean",
              ]}
            />
          </Panel>
          <Panel title="Hospital fold results">
            <label className="filter">
              Hospital{" "}
              <select
                value={hospital}
                onChange={(e) => setHospital(e.target.value)}
              >
                {["All", ...new Set(d.results.map((x) => x.hospital))].map(
                  (x) => (
                    <option key={x}>{x}</option>
                  ),
                )}
              </select>
            </label>
            <Table
              rows={d.results.filter(
                (x) => hospital === "All" || x.hospital === hospital,
              )}
              columns={[
                "model",
                "hospital",
                "test_samples",
                "roc_auc",
                "recall",
                "specificity",
                "tn",
                "fp",
                "fn",
                "tp",
              ]}
            />
          </Panel>
        </>
      )}
    </State>
  );
}

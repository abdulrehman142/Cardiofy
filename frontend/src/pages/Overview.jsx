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
export default function Overview({ navigate }) {
  const metrics = useData("/metrics"),
    analytics = useData("/analytics"),
    features = useData("/features");
  return (
    <>
      <div className="hero">
        <div>
          <span className="eyebrow">CARDIOVASCULAR MACHINE LEARNING</span>
          <h2>
            A clearer view of
            <br />
            every prediction.
          </h2>
          <p>
            Explore model behavior, understand contributing features,
            <br />
            and examine the evidence behind the results.
          </p>
          <button onClick={() => navigate("Prediction lab")}>
            Open prediction lab <ChevronRight size={17} />
          </button>
        </div>
        <div className="heart-art">
          <HeartPulse strokeWidth={0.9} />
          <span>Research. Explain. Evaluate.</span>
        </div>
      </div>
      <State state={metrics}>
        {(d) => (
          <>
            <Stats
              items={[
                [
                  "Test ROC-AUC",
                  pct(d.metadata.metrics.roc_auc),
                  "Held-out test partition",
                ],
                [
                  "Test recall",
                  pct(d.metadata.metrics.recall),
                  "Positive cases detected",
                ],
                [
                  "Selected model",
                  d.metadata.model_name,
                  `${d.metadata.calibration} calibration · v${d.metadata.model_version}`,
                ],
                [
                  "CV ROC-AUC",
                  pct(d.metadata.selection_cv_roc_auc),
                  "Model selection · training only",
                ],
              ]}
            />
            <div className="notice">
              <CircleAlert size={20} />
              <span>{d.limitation}</span>
            </div>
          </>
        )}
      </State>
      <div className="grid two">
        <Panel
          title="What drives the model"
          subtitle="Global mean absolute SHAP contribution"
        >
          <State state={features}>
            {(d) => (
              <Bars
                data={d
                  .slice(0, 6)
                  .map((x) => ({ name: x.feature, value: x.mean_abs_shap }))}
              />
            )}
          </State>
        </Panel>
        <Panel
          title="The research cohort"
          subtitle="Processed UCI heart disease dataset"
        >
          <State state={analytics}>
            {(d) => (
              <>
                <div className="cohort-total">
                  {d.total_records.toLocaleString()}{" "}
                  <span>records across {d.total_hospitals} sources</span>
                </div>
                {d.hospital_summary.map((x) => (
                  <div className="source" key={x.hospital}>
                    <div>
                      <span>{x.hospital}</span>
                      <b>{x.records}</b>
                    </div>
                    <div className="track">
                      <i
                        style={{
                          width: `${(x.records / d.total_records) * 100}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </>
            )}
          </State>
        </Panel>
      </div>
    </>
  );
}

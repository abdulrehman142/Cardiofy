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
export default function Dataset() {
  const state = useData("/analytics");
  return (
    <State state={state}>
      {(d) => (
        <>
          <Stats
            items={[
              ["Records", d.total_records, "Processed dataset"],
              ["Features", d.total_features, "Model input schema"],
              ["Hospital sources", d.total_hospitals, "Retrospective cohort"],
              [
                "Positive labels",
                pct(d.disease_rate),
                `${d.disease_cases} records`,
              ],
            ]}
          />
          <div className="grid two">
            <Panel
              title="Age distribution"
              subtitle="Record counts by age interval"
            >
              <Bars
                data={Object.entries(d.age_distribution).map(
                  ([name, value]) => ({ name, value }),
                )}
              />
            </Panel>
            <Panel
              title="Missing input values"
              subtitle="Before pipeline imputation"
            >
              <Bars
                data={Object.entries(d.missing_values).map(([name, value]) => ({
                  name,
                  value,
                }))}
              />
            </Panel>
          </div>
          <Panel title="Hospital cohort summary">
            <Table
              rows={d.hospital_summary}
              columns={["hospital", "records", "disease_cases", "disease_rate"]}
            />
          </Panel>
        </>
      )}
    </State>
  );
}

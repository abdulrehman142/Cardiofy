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
export default function HistoryPage() {
  const state = useData("/history"),
    [confirm, setConfirm] = useState(false),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  async function clear() {
    setBusy(true);
    try {
      await api("/history", { method: "DELETE" });
      setConfirm(false);
      state.refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Panel
      title="Local prediction history"
      subtitle="Opt-in results only. No patient inputs or identifiers are stored."
    >
      <div className="controls">
        <button onClick={state.refresh}>
          <RefreshCw size={15} />
          Refresh
        </button>
        {confirm ? (
          <span>
            Delete all saved results?{" "}
            <button disabled={busy} className="danger" onClick={clear}>
              Delete all
            </button>{" "}
            <button onClick={() => setConfirm(false)}>Cancel</button>
          </span>
        ) : (
          <button className="danger" onClick={() => setConfirm(true)}>
            Clear history
          </button>
        )}
      </div>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      <State state={state}>
        {(d) => (
          <Table
            rows={d.map((x) => ({
              ...x,
              probability: pct(x.probability),
              prediction: x.prediction ? "Positive label" : "Negative label",
            }))}
            columns={[
              "id",
              "prediction",
              "probability",
              "model_name",
              "model_version",
              "created_at",
            ]}
          />
        )}
      </State>
    </Panel>
  );
}

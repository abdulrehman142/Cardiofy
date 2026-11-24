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
function Runs({ id }) {
  const state = useData(`/experiments/${id}/runs`),
    [selected, setSelected] = useState(null);
  useEffect(() => setSelected(null), [id]);
  return (
    <State state={state}>
      {(d) => (
        <>
          <p className="muted">
            Up to 100 most recent runs. Select a run to inspect its metrics and
            parameters.
          </p>
          <div className="runs">
            {d.map((x) => (
              <button
                key={x.run_id}
                className={selected?.run_id === x.run_id ? "selected" : ""}
                onClick={() => setSelected(x)}
              >
                <span>
                  <b>{x.run_name || x.run_id}</b>
                  <small>
                    {x.start_time
                      ? new Date(x.start_time).toLocaleString()
                      : "No timestamp"}
                  </small>
                </span>
                <span className="pill">{x.status}</span>
              </button>
            ))}
          </div>
          {!d.length && <p className="empty">No runs in this experiment.</p>}
          {selected && (
            <div className="grid two">
              <Panel title="Logged metrics">
                <Table
                  rows={Object.entries(selected.metrics).map(
                    ([key, value]) => ({ key, value }),
                  )}
                  columns={["key", "value"]}
                />
              </Panel>
              <Panel title="Parameters">
                <Table
                  rows={Object.entries(selected.parameters).map(
                    ([key, value]) => ({ key, value }),
                  )}
                  columns={["key", "value"]}
                />
              </Panel>
            </div>
          )}
        </>
      )}
    </State>
  );
}
export default function Experiments() {
  const state = useData("/experiments"),
    [id, setId] = useState("");
  return (
    <Panel
      title="MLflow experiment explorer"
      subtitle="Read-only access to the existing local SQLite tracking store"
    >
      <State state={state}>
        {(d) =>
          d.length ? (
            <>
              <label className="filter">
                Experiment{" "}
                <select
                  value={id || d[0].id}
                  onChange={(e) => setId(e.target.value)}
                >
                  {d.map((x) => (
                    <option key={x.id} value={x.id}>
                      {x.name}
                    </option>
                  ))}
                </select>
              </label>
              <Runs id={id || d[0].id} />
            </>
          ) : (
            <p className="empty">No experiments available.</p>
          )
        }
      </State>
    </Panel>
  );
}

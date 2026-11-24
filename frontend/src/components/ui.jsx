import React, { useEffect, useState } from "react";
import { CircleAlert } from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from "recharts";
import { api } from "../api";
export const pct = (n) =>
  n == null ? "—" : `${(Number(n) * 100).toFixed(1)}%`;
export const fmt = (n) =>
  n == null
    ? "—"
    : typeof n === "number"
      ? Number.isInteger(n)
        ? n.toLocaleString()
        : n.toFixed(4)
      : String(n);
export function useData(path) {
  const [state, set] = useState({ loading: true }),
    [tick, refresh] = useState(0);
  useEffect(() => {
    let active = true;
    set({ loading: true });
    api(path)
      .then((data) => active && set({ data }))
      .catch((e) => active && set({ error: e.message }));
    return () => {
      active = false;
    };
  }, [path, tick]);
  return { ...state, refresh: () => refresh((x) => x + 1) };
}
export function State({ state, children }) {
  return state.loading ? (
    <div className="empty" role="status">
      Loading local research artifacts…
    </div>
  ) : state.error ? (
    <div className="error" role="alert">
      <CircleAlert size={22} />
      <div>
        {state.error}
        <br />
        <button onClick={state.refresh}>Try again</button>
      </div>
    </div>
  ) : (
    children(state.data)
  );
}
export function Panel({ title, subtitle, children }) {
  return (
    <section className="panel">
      <div className="panel-title">
        <h2>{title}</h2>
        {subtitle && <p>{subtitle}</p>}
      </div>
      {children}
    </section>
  );
}
export function Table({ rows, columns }) {
  return !rows?.length ? (
    <div className="empty">No records available.</div>
  ) : (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c}>{c.replaceAll("_", " ")}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={row.run_id || row.id || i}>
              {columns.map((c) => (
                <td key={c}>{fmt(row[c])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
export function Bars({
  data,
  keys = ["value"],
  name = "name",
  percent = false,
}) {
  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 12, right: 12, left: 0, bottom: 24 }}
        >
          <CartesianGrid strokeDasharray="4 4" vertical={false} />
          <XAxis
            dataKey={name}
            tick={{ fontSize: 12 }}
            interval={0}
            angle={-12}
            textAnchor="end"
          />
          <YAxis tickFormatter={percent ? pct : undefined} width={55} />
          <Tooltip formatter={(v) => (percent ? pct(v) : fmt(v))} />
          {keys.length > 1 && <Legend />}
          {keys.map((k, i) => (
            <Bar
              key={k}
              dataKey={k}
              fill={["#277c68", "#96cdbb", "#98a7d5"][i % 3]}
              radius={[5, 5, 0, 0]}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
export function Stats({ items }) {
  return (
    <div className="stats">
      {items.map(([label, value, note], i) => (
        <article className="stat" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
          <small>{note}</small>
          <div className={`stat-line s${i}`} />
        </article>
      ))}
    </div>
  );
}

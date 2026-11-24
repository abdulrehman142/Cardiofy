import React, { useState } from "react";
import { Activity, ShieldCheck, RotateCcw, LoaderCircle } from "lucide-react";
import { api } from "../api";
import { pct } from "../components/ui";

const fields = [
  ["age", "Age", "years", 1, 120, "e.g. 55"],
  ["trestbps", "Resting blood pressure", "mm Hg", 1, 350, "e.g. 140"],
  ["chol", "Cholesterol", "mg/dL", 1, 1000, "e.g. 240"],
  ["thalch", "Maximum heart rate", "bpm", 1, 300, "e.g. 150"],
  ["oldpeak", "ST depression", "oldpeak", -10, 15, "e.g. 1.5"],
  ["ca", "Major vessels", "0–3", 0, 3, "Select a value"],
];
const blank = () => Object.fromEntries(fields.map(([key]) => [key, ""]));

export default function Prediction() {
  const [form, setForm] = useState(blank);
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [snapshot, setSnapshot] = useState(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  function change(key, value) {
    setForm((current) => ({ ...current, [key]: value }));
    setResult(null);
    setExplanation(null);
    setSnapshot(null);
    setError("");
  }
  function reset() {
    setForm(blank());
    setResult(null);
    setExplanation(null);
    setSnapshot(null);
    setError("");
  }
  async function predict(event) {
    event.preventDefault();
    setBusy("predict");
    setError("");
    setResult(null);
    setExplanation(null);
    const patient = Object.fromEntries(
      fields.map(([key]) => [key, form[key] === "" ? null : Number(form[key])]),
    );
    try {
      const data = await api("/predict", {
        method: "POST",
        body: JSON.stringify({ ...patient, save_history: false }),
      });
      setResult(data);
      setSnapshot(patient);
    } catch (error) {
      setError(error.message);
    } finally {
      setBusy("");
    }
  }
  async function explain() {
    setBusy("explain");
    setError("");
    try {
      setExplanation(
        await api("/explain", {
          method: "POST",
          body: JSON.stringify(snapshot),
        }),
      );
    } catch (error) {
      setError(error.message);
    } finally {
      setBusy("");
    }
  }
  return (
    <>
      <section className="intro">
        <h1>Heart disease prediction</h1>
        <p>Enter your measurements to get an experimental prediction.</p>
      </section>
      <div className="prediction-layout">
        <section className="input-section" aria-labelledby="measurements-title">
          <div className="section-heading">
            <div>
              <h2 id="measurements-title">Your measurements</h2>
              <p>Start with age. Add the details you have.</p>
            </div>
            <button
              className="text-button reset"
              onClick={reset}
              disabled={!!busy}
            >
              <RotateCcw size={15} /> Reset
            </button>
          </div>
          <form onSubmit={predict}>
            <div className="form-grid">
              {fields.map(([key, label, unit, min, max, placeholder]) => (
                <label className="field" key={key}>
                  <span className="field-label">
                    {label}
                    {key === "age" && <span className="required"> *</span>}
                  </span>
                  <div className="input-wrap">
                    {key === "ca" ? (
                      <select
                        aria-label="Major vessels"
                        value={form[key]}
                        disabled={!!busy}
                        onChange={(event) => change(key, event.target.value)}
                      >
                        <option value="">Not available</option>
                        {[0, 1, 2, 3].map((value) => (
                          <option key={value} value={value}>
                            {value}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <>
                        <input
                          type="number"
                          aria-label={label}
                          step={key === "age" ? "1" : "any"}
                          min={min}
                          max={max}
                          required={key === "age"}
                          value={form[key]}
                          onChange={(event) => change(key, event.target.value)}
                          disabled={!!busy}
                          placeholder={placeholder}
                        />
                        <span className="input-unit">{unit}</span>
                      </>
                    )}
                  </div>
                </label>
              ))}
            </div>
            <p className="form-hint">
              Only age is required. Missing values use the model’s learned
              defaults.
            </p>
            <button className="primary predict-button" disabled={!!busy}>
              {busy === "predict" ? (
                <LoaderCircle className="spin" size={18} />
              ) : (
                <Activity size={18} />
              )}
              {busy === "predict" ? "Calculating…" : "Get prediction"}
            </button>
            <div className="privacy-note">
              <ShieldCheck size={16} />
              <span>Processed locally. Your measurements aren’t saved.</span>
            </div>
          </form>
        </section>
        <div className="result-column">
          <section
            className={`result-section ${result ? "has-result" : ""}`}
            aria-live="polite"
            aria-busy={!!busy}
          >
            <h2>Prediction result</h2>
            {result ? (
              <>
                <p className="result-label">Positive-label probability</p>
                <div className="result-probability">
                  {pct(result.probability)}
                </div>
                <p className="result-description">
                  {result.prediction ? "Above" : "Below"} the{" "}
                  {pct(result.threshold)} classification threshold.
                </p>
                <div className="result-divider" />
                <div className="model-caption">
                  <span>
                    {result.model_name} · v{result.model_version}
                  </span>
                  <span>Experimental</span>
                </div>
                {!explanation && (
                  <button
                    className="explain-button"
                    disabled={!!busy}
                    onClick={explain}
                  >
                    {busy === "explain" ? (
                      <LoaderCircle size={17} className="spin" />
                    ) : (
                      <Activity size={17} />
                    )}
                    {busy === "explain"
                      ? "Finding contributing factors…"
                      : "Explain result"}
                  </button>
                )}
              </>
            ) : (
              <div className="result-placeholder">
                <Activity size={30} strokeWidth={1.4} />
                <p>Your prediction will appear here.</p>
                <span>Enter measurements, then select Get prediction.</span>
              </div>
            )}
          </section>
          {error && (
            <div className="error" role="alert">
              {error}
            </div>
          )}
          {explanation && (
            <section className="explanation-section">
              <h2>Behind the result</h2>
              <p>Top contributions, in probability percentage points.</p>
              {explanation.contributions
                .filter((item) => Math.abs(item.shap_value) > 1e-8)
                .slice(0, 6)
                .map((item) => (
                  <div className="contribution" key={item.feature}>
                    <div>
                      <span>{item.label}</span>
                      <b
                        className={
                          item.shap_value > 0 ? "positive" : "negative"
                        }
                      >
                        {item.impact_percentage_points > 0 ? "+" : ""}
                        {item.impact_percentage_points.toFixed(2)} pp
                      </b>
                    </div>
                    <div className="track">
                      <i
                        className={item.shap_value > 0 ? "positive-bg" : ""}
                        style={{
                          width: `${Math.min(100, Math.abs(item.shap_value) * 300)}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              <p className="explanation-note">
                Describes model behavior, not cause and effect. Starting
                probability: {pct(explanation.base_probability)}.
              </p>
            </section>
          )}
        </div>
      </div>
      <details className="model-note">
        <summary>About this model & its limitations</summary>
        <div>
          <p>
            This experimental model uses the six measurements shown above. Seven
            other categorical features were dropped during its original
            training, so they cannot influence its predictions. Repairing that
            limitation requires retraining and revalidation.
          </p>
          <p>
            Age is required; other missing values are imputed by the fitted
            pipeline. Input limits are software bounds, not clinical reference
            ranges. Neither predictions nor explanations establish clinical
            effectiveness.
          </p>
        </div>
      </details>
    </>
  );
}

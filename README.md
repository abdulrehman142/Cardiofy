# Cardiofy

Local experimental heart disease ML workspace: React/Vite frontend, modular FastAPI backend, existing calibrated SVM and permutation SHAP explanations. Not a medical diagnosis or clinically validated system.

## Run locally

From the project root (actual path `/Users/abdulrehman/Desktop/Cardiofy`):

```sh
.venv/bin/python -m pip install -r backend/requirements.txt
.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```sh
cd /Users/abdulrehman/Desktop/Cardiofy/frontend
npm ci
npm run dev
```

Open http://127.0.0.1:5173. API documentation: http://127.0.0.1:8000/docs.

The shared frontend Axios client defaults to the relative `/api` URL in development and production, with a 180,000 ms timeout for explanations. Set `VITE_API_URL` at build time only when using a separate API origin (include its `/api` path). Without an override, the production web server must proxy `/api` to FastAPI. Existing page helpers use this same client; new code can import its default export and call `.get()` or `.post()`.

Keep both services bound to loopback. Vite proxies `/api` to port 8000. Preview uses the same port 5173 and proxy; stop the dev server before `npm run preview`. No external font, analytics or inference service is used. This is a single-user local prototype, without multi-user authentication; do not expose it on a network.

## Features and source layout

- `frontend/src/`: a focused website with horizontal navigation for Prediction and Model Comparison. Predictions use the six active model inputs, with SHAP explanations available on demand. Baseline and tuned results appear in a single metric-selectable comparison table. Legacy research page sources remain preserved but are not imported into the website. All displayed results use real API requests.
- `backend/app/config.py`, `schemas.py`, `database.py`: root-relative artifact paths, constrained inputs and opt-in SQLite persistence.
- `backend/app/routes/`: prediction/explanation, analytics/reports, experiments and history routes.
- `backend/app/services/`: cached model/SHAP loading and CSV/metadata adapters.
- `tests/test_api.py`: real artifact integration checks, model parity, SHAP additivity, validation, opt-in history and local security.
- Existing `ml/`, data, models, DVC, notebooks, reports and MLflow artifacts remain in their original locations. No retraining, report regeneration, DVC add, commit or deployment is performed by the app.

## Existing model limitation

The final artifact is SVM v1.0.0, baseline selection, sigmoid calibration, decision threshold 0.5. Its numerical feature name is **`thalch`**, not `thalach` (the API accepts `thalach` as a legacy alias). The dataset hospital column is `dataset`.

The current training/evaluation code converted text categories with `pd.to_numeric(errors="coerce")`. Inspection of the saved imputer statistics confirms seven entirely empty categorical columns: `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `thal`. They were dropped during fitting. Only age, blood pressure, cholesterol, maximum heart rate, ST depression and `ca` affect the saved model. The app documents this limitation in expandable model notes, retains the artifact faithfully, and displays only active inputs. Correcting this requires deliberate retraining, revalidation and new versioned artifacts; changing inference encodings alone cannot repair it.

The numerical bounds in the form/API are software input bounds, not clinical reference ranges. Optional missing inputs use the already-fitted pipeline imputation. SHAP describes the existing model, not clinical causation. The explainer uses the original training split and numerical coercion to match the fitted artifact. Cross-validation, held-out test, and leave-one-hospital-out results remain separately labeled.

## Privacy and local defaults

History is off by default. `/api/predict` saves only when `save_history` is true; `/api/explain` never saves. SQLite stores only label, probability, model name/version and UTC timestamp. No patient identifiers or raw inputs are stored. The simplified prediction UI does not save history. Existing history endpoints remain available through the local API. Set `CARDIOFY_HISTORY_DB` to use a different history database.

Allowed browser origins are localhost/127.0.0.1 on port 5173. Host validation and mutation-origin checks protect local endpoints. API responses use `Cache-Control: no-store`. Only allowlisted evaluation and SHAP PNGs can be served; datasets, test predictions and arbitrary reports are not mounted publicly.

Experiments use parameterized **read-only SQLite** queries against the existing MLflow database, including latest metrics and parameters. This deliberately avoids SDK initialization/migrations or global tracking-URI changes. It supports the current local schema; another store/schema produces an explicit unavailable response. Run limits are bounded (default 100, max 500).

## API

All routes have `/api` prefix:

- `GET /health`, `/models`, `/metrics`, `/comparison`, `/hospitals`, `/features`, `/analytics`
- `POST /predict`, `/explain` (age required, other model features optional, extra fields rejected)
- `GET /experiments`, `/experiments/{id}/runs?limit=100`
- `GET /history?limit=100`, `DELETE /history`
- `GET /plots/{evaluation|shap}/{allowlisted-name}.png`

Missing datasets/reports are reported explicitly; missing/unloadable model artifacts give unavailable responses. Restore absent tracked artifacts through the existing DVC workflow. Do not load untrusted joblib files.

## Verify

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ml.test_prediction
.venv/bin/python -m ml.test_explanation
cd frontend
npm run build
```

`pytest.ini` scopes test discovery to `tests/`; legacy ML demonstration scripts are run explicitly because they execute at import time. SHAP report generation is intentionally not run during app testing because it overwrites existing reports and adds MLflow runs.

Tested with the existing Python 3.11 environment, FastAPI 0.165.0, Pydantic 2.14.4, scikit-learn 1.9.1, Node 22.18.0 and npm 10.9.3. Frontend dependency versions are captured in `frontend/package-lock.json`. The original ML requirements remain unchanged. The saved estimator emits warnings about its dropped categorical columns; the existing TestClient and SHAP dependencies also emit deprecation warnings.

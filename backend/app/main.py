import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from backend.app.config import DISCLAIMER, REPORT_DIR
from backend.app.database import init_db
from backend.app.services.model_service import get_model_service
from backend.app.routes import prediction, analytics, experiments, history

@asynccontextmanager
async def lifespan(app):
    init_db()
    yield

app = FastAPI(title="Cardiofy API", version="1.0.0", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:3000", "http://127.0.0.1:3000",
]
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type"])

@app.middleware("http")
async def local_security(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin and origin not in ORIGINS:
        return JSONResponse({"detail":"Origin is not allowed"}, status_code=403)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

@app.exception_handler(Exception)
async def unavailable(request, exc):
    logging.getLogger("cardiofy").error("Request failed: %s", type(exc).__name__)
    return JSONResponse({"detail":"Required local artifact or dependency unavailable. Check the server and project setup."}, status_code=503)

for router in (prediction.router, analytics.router, experiments.router, history.router):
    app.include_router(router, prefix="/api")

@app.get("/api/health")
def health():
    try:
        service = get_model_service()
        return {"status":"ok", "model":service.metadata["model_name"], "disclaimer":DISCLAIMER}
    except Exception as exc:
        raise HTTPException(503, "Saved model unavailable") from exc

PLOTS = {f"evaluation/{name}.png" for name in ["confusion_matrix", "roc_curve", "precision_recall_curve", "calibration_curve"]}
PLOTS |= {f"shap/{name}.png" for name in ["shap_beeswarm", "shap_feature_importance", "shap_waterfall", "shap_heatmap"]}

@app.get("/api/plots/{folder}/{name}")
def plot(folder: str, name: str):
    relative = f"{folder}/{name}"
    if relative not in PLOTS or not (REPORT_DIR / relative).is_file():
        raise HTTPException(404, "Plot unavailable")
    return FileResponse(REPORT_DIR / relative, media_type="image/png")

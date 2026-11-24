import json
import pandas as pd
from fastapi import APIRouter, HTTPException
from backend.app.config import DATA_PATH, MODEL_LIMITATION
from backend.app.services.report_service import read_report, metadata
router = APIRouter(tags=["Analytics"])

@router.get("/models")
def models():
    return {"baseline": read_report("training/baseline_results.csv"), "tuned": read_report("tuning/tuned_results.csv"), "selected": metadata()["model_name"]}

@router.get("/metrics")
def metrics():
    return {"metadata": metadata(), "limitation": MODEL_LIMITATION, "calibration": read_report("evaluation/calibration_comparison.csv"), "classification": read_report("evaluation/classification_report.csv")}

@router.get("/comparison")
def comparison():
    return read_report("comparison/baseline_vs_tuned.csv")

@router.get("/hospitals")
def hospitals():
    return {"results": read_report("hospital_validation/hospital_results.csv"), "summary": read_report("hospital_validation/hospital_summary.csv")}

@router.get("/features")
def features():
    rows = read_report("shap/feature_importance.csv")
    for row in rows:
        row["label"] = row.get("label") or ("Maximum Heart Rate" if row["feature"] == "thalch" else row["feature"])
    return rows

@router.get("/analytics")
def analytics():
    if not DATA_PATH.is_file():
        raise HTTPException(404, "Processed dataset unavailable. Restore it with DVC.")
    df = pd.read_csv(DATA_PATH)
    hospitals = df.groupby("dataset").agg(records=("target", "size"), disease_cases=("target", "sum"), disease_rate=("target", "mean")).reset_index().rename(columns={"dataset":"hospital"})
    bins = pd.cut(df.age, bins=[0,30,40,50,60,70,120], right=False).value_counts(sort=False)
    return {"total_records": len(df), "total_features": len(metadata()["features"]), "total_hospitals": int(df.dataset.nunique()), "disease_cases": int(df.target.sum()), "disease_rate": float(df.target.mean()), "hospital_distribution": df.dataset.value_counts().to_dict(), "class_distribution": df.target.value_counts().to_dict(), "missing_values": df[metadata()["features"]].isna().sum().to_dict(), "hospital_summary": json.loads(hospitals.to_json(orient="records")), "age_distribution": {str(k):int(v) for k,v in bins.items()}}

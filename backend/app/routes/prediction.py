from threading import Lock
from fastapi import APIRouter
from backend.app.schemas import PatientInput, PredictionResponse
from backend.app.services.model_service import get_model_service, get_explainer
from backend.app.database import save_prediction
from backend.app.config import MODEL_LIMITATION
router = APIRouter(tags=["Predictions"])
explanation_lock = Lock()

@router.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    result = get_model_service().predict(patient.model_dump(exclude={"save_history"}))
    if patient.save_history:
        save_prediction(result)
    return result

@router.post("/explain")
def explain(patient: PatientInput):
    with explanation_lock:
        result = get_explainer().explain_patient(patient.model_dump(exclude={"save_history"}))
    result["limitation"] = MODEL_LIMITATION
    return result

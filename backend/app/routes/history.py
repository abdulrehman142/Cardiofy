from fastapi import APIRouter, Query
from backend.app.database import get_predictions, clear_predictions
router = APIRouter(tags=["History"])

@router.get("/history")
def history(limit: int = Query(100, ge=1, le=500)):
    return get_predictions(limit)

@router.delete("/history")
def delete_history():
    clear_predictions()
    return {"message":"Prediction history cleared."}

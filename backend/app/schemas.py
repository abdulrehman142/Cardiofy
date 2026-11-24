from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, AliasChoices

class PatientInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    age: float = Field(gt=0, le=120)
    trestbps: float | None = Field(None, gt=0, le=350)
    chol: float | None = Field(None, gt=0, le=1000)
    thalch: float | None = Field(None, gt=0, le=300, validation_alias=AliasChoices("thalch", "thalach"))
    oldpeak: float | None = Field(None, ge=-10, le=15)
    sex: Literal[0, 1] | None = None
    cp: Literal[1, 2, 3, 4] | None = None
    fbs: Literal[0, 1] | None = None
    restecg: Literal[0, 1, 2] | None = None
    exang: Literal[0, 1] | None = None
    slope: Literal[1, 2, 3] | None = None
    ca: Literal[0, 1, 2, 3] | None = None
    thal: Literal[3, 6, 7] | None = None
    save_history: bool = False

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    threshold: float
    model_name: str
    model_version: str
    disclaimer: str
    limitation: str

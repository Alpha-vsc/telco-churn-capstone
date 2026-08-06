"""
api/main.py — API REST de scoring du churn (Mission 5).

Lancer : uvicorn api.main:app --reload --port 8000
Docs interactives : http://localhost:8000/docs
"""
import json
import os
import sys
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import MODEL_VERSION, DECISION_THRESHOLD, EXPECTED_FEATURES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.joblib")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "metadata.json")

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Prédit la probabilité de résiliation d'un client télécom.",
    version=MODEL_VERSION,
)

_model = None
_metadata = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Modèle introuvable à {MODEL_PATH}. Lancer `python src/train.py` d'abord.")
        _model = joblib.load(MODEL_PATH)
    return _model


def get_metadata():
    global _metadata
    if _metadata is None:
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)
    return _metadata


class ClientFeatures(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100, description="Ancienneté en mois")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
                "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
                "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 93.85, "TotalCharges": 187.70,
            }
        }


class PredictionResponse(BaseModel):
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float
    decision_threshold: float
    model_version: str


@app.get("/health")
def health():
    """Statut de l'API et version du modèle chargé."""
    try:
        get_model()
        status = "ok"
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    return {"status": status, "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientFeatures):
    """Prédit la probabilité de churn d'un client à partir de ses caractéristiques."""
    model = get_model()
    row = pd.DataFrame([client.model_dump()])[EXPECTED_FEATURES]
    try:
        proba = float(model.predict_proba(row)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erreur de scoring : {e}")

    return PredictionResponse(
        churn_prediction="Yes" if proba >= DECISION_THRESHOLD else "No",
        churn_probability=round(proba, 4),
        decision_threshold=DECISION_THRESHOLD,
        model_version=MODEL_VERSION,
    )


@app.get("/model-info")
def model_info():
    """Features attendues, métadonnées d'entraînement, performance de validation."""
    meta = get_metadata()
    return {
        "model_version": meta["model_version"],
        "expected_features": meta["expected_features"],
        "decision_threshold": meta["decision_threshold"],
        "cost_model_eur": meta["cost_model_eur"],
        "reference_metrics_cv_train": meta["reference_metrics_cv_train"],
        "test_set_metrics": meta["test_set_metrics"],
        "trained_on": meta["trained_on"],
    }

"""
src/model.py — Configuration finale du modèle champion (issue de la Mission 4).

Source unique de vérité pour : l'entraînement (train.py), les tests (tests/), et
l'API (api/main.py). Toute évolution du modèle (nouveau tuning, nouveau seuil) ne
doit être modifiée qu'ici.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from src.pipeline import build_pipeline

MODEL_VERSION = "1.0.0"

# Meilleurs hyperparamètres — Optuna, Mission 4 (60 essais, pruner médian, CV 3-fold)
BEST_PARAMS = dict(
    C=2.4806577069936093,
    penalty="elasticnet",
    l1_ratio=0.48060749631107325,
    class_weight="balanced",
    tol=0.0011419305693886833,
    max_iter=1000,
    fit_intercept=True,
)
RANDOM_STATE = 42

# Seuil de décision optimisé sur le coût métier — Mission 4
DECISION_THRESHOLD = 0.18

# Modèle de coût métier — Mission 0
COST_FALSE_NEGATIVE_EUR = 175  # churner manqué
COST_FALSE_POSITIVE_EUR = 47  # offre envoyée à un client fidèle

# Métadonnées de performance de référence (CV 5-fold, train — Mission 4)
REFERENCE_METRICS = {
    "f2_cv_mean": 0.7221,
    "f2_cv_std": 0.0290,
    "precision_at_threshold": 0.457,
    "recall_at_threshold": 0.892,
    "brier_score_calibrated": 0.1353,
}

EXPECTED_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]


def build_champion_classifier() -> LogisticRegression:
    return LogisticRegression(solver="saga", random_state=RANDOM_STATE, **BEST_PARAMS)


def build_final_pipeline(calibrated: bool = True):
    """Construit le pipeline final : preprocessing (sans fuite) + modèle champion,
    calibré (Platt/sigmoid) par défaut — c'est la version déployée en production."""
    base_pipe = build_pipeline(build_champion_classifier())
    if not calibrated:
        return base_pipe
    return CalibratedClassifierCV(base_pipe, method="sigmoid", cv=3)

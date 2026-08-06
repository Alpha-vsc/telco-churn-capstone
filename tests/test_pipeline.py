"""
tests/test_pipeline.py — Suite de tests du pipeline de production (Mission 5).

Lancer avec : pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
import pandas as pd
import pytest

from src.model import EXPECTED_FEATURES, DECISION_THRESHOLD

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.joblib")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Telco-Customer-Churn.csv")


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def sample_data():
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=["customerID", "Churn"]).head(20)


def test_output_shape(model, sample_data):
    """La sortie a la bonne forme : une prédiction par ligne d'entrée."""
    proba = model.predict_proba(sample_data)
    assert proba.shape == (len(sample_data), 2)
    preds = model.predict(sample_data)
    assert preds.shape == (len(sample_data),)


def test_probabilities_in_valid_range(model, sample_data):
    """Les probabilités prédites sont bien dans [0, 1] et somment à 1 par ligne."""
    proba = model.predict_proba(sample_data)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_pipeline_handles_missing_values(model, sample_data):
    """Le pipeline gère les valeurs manquantes (imputation) sans lever d'exception,
    aussi bien sur une colonne numérique que catégorielle."""
    corrupted = sample_data.copy()
    corrupted.loc[corrupted.index[0], "MonthlyCharges"] = np.nan
    corrupted.loc[corrupted.index[1], "InternetService"] = np.nan
    corrupted.loc[corrupted.index[2], "TotalCharges"] = np.nan

    proba = model.predict_proba(corrupted)
    assert not np.any(np.isnan(proba)), "Le pipeline a produit des NaN en sortie"


def test_expected_features_present(sample_data):
    """Les features attendues par le modèle sont bien celles documentées dans
    src/model.py (contrat d'interface entre le modèle et l'API)."""
    assert set(EXPECTED_FEATURES) == set(sample_data.columns), (
        "Le schéma des features d'entrée a divergé de EXPECTED_FEATURES — "
        "src/model.py doit être mis à jour si le schéma de données change."
    )


def test_missing_feature_raises_informative_error(model, sample_data):
    """Retirer une feature attendue doit faire échouer la prédiction de façon
    explicite (pas un résultat silencieusement erroné)."""
    incomplete = sample_data.drop(columns=["Contract"])
    with pytest.raises(Exception):
        model.predict_proba(incomplete)


def test_performance_on_reference_set():
    """La performance sur un mini-jeu de référence (echantillon stratifié fixe)
    reste correcte — garde-fou contre une régression silencieuse du modèle."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import fbeta_score

    df = pd.read_csv(DATA_PATH)
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["customerID", "Churn"])
    _, X_ref, _, y_ref = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = joblib.load(MODEL_PATH)
    proba = model.predict_proba(X_ref)[:, 1]
    preds = (proba >= DECISION_THRESHOLD).astype(int)
    f2 = fbeta_score(y_ref, preds, beta=2)

    assert f2 >= 0.60, f"F2 sur le jeu de référence ({f2:.4f}) sous le seuil d'alerte de 0.60"


def test_reload_gives_identical_predictions(sample_data):
    """Un modèle rechargé depuis le disque donne des prédictions strictement
    identiques au modèle original — condition nécessaire pour un déploiement fiable."""
    m1 = joblib.load(MODEL_PATH)
    m2 = joblib.load(MODEL_PATH)
    proba1 = m1.predict_proba(sample_data)
    proba2 = m2.predict_proba(sample_data)
    np.testing.assert_array_equal(proba1, proba2)


def test_high_risk_profile_scores_above_low_risk_profile(model):
    """Sanity check métier : un profil archétypal à haut risque (nouveau client,
    contrat mensuel, fibre chère) doit recevoir un score plus élevé qu'un profil
    archétypal fidèle (ancienneté maximale, contrat 2 ans, sans internet)."""
    high_risk = pd.DataFrame([{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 95.0,
    }])
    low_risk = pd.DataFrame([{
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
        "tenure": 72, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "No", "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service", "DeviceProtection": "No internet service",
        "TechSupport": "No internet service", "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service", "Contract": "Two year",
        "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 20.0, "TotalCharges": 1440.0,
    }])
    proba_high = model.predict_proba(high_risk)[0, 1]
    proba_low = model.predict_proba(low_risk)[0, 1]
    assert proba_high > proba_low
    assert proba_high > 0.5
    assert proba_low < 0.2

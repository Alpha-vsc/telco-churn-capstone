"""
src/features.py — Transformers scikit-learn custom pour le projet Churn Telco.

Principe directeur (Mission 2) : toute transformation qui APPREND une statistique
(moyenne, médiane, modalité la plus fréquente, encodage cible...) doit vivre dans
un pipeline fit() sur le train uniquement. Les transformations ci-dessous qui
n'apprennent RIEN du jeu de données (règles déterministes, seuils fixes) peuvent
être placées avant le split sans risque de fuite — mais sont volontairement gardées
dans le pipeline pour obtenir un objet unique, reproductible, et sérialisable.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


class TotalChargesFixer(BaseEstimator, TransformerMixin):
    """Corrige TotalCharges : les 11 valeurs vides correspondent exactement aux
    clients avec tenure == 0 (nouveaux clients non encore facturés). Ce n'est pas
    un manquant statistique : c'est un zéro logique. Règle fixe, ne rien apprendre
    du train -> aucun risque de fuite même si appliqué avant le split.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        X.loc[X["tenure"] == 0, "TotalCharges"] = 0.0
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Ajoute 3 features dérivées, chacune validée par gain de CV (Mission 2) :
      - tenure_group       : ancienneté catégorisée (bucket fixe, ne s'apprend pas)
      - n_services          : nombre de services actifs souscrits (compte déterministe)
      - charges_per_tenure  : TotalCharges / (tenure + 1), remplace la redondance
                               tenure / MonthlyCharges / TotalCharges (r=0.9996, cf M1)

    Aucun de ces calculs n'utilise de statistique apprise sur le train -> peut être
    fit() n'importe où sans fuite, mais reste dans le pipeline pour la reproductibilité.
    """

    def __init__(self, add_tenure_group=True, add_n_services=True, add_charges_ratio=True):
        self.add_tenure_group = add_tenure_group
        self.add_n_services = add_n_services
        self.add_charges_ratio = add_charges_ratio

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.add_tenure_group:
            X["tenure_group"] = pd.cut(
                X["tenure"],
                bins=[-0.1, 12, 24, 48, 72],
                labels=["0-12m", "13-24m", "25-48m", "49-72m"],
            ).astype(str)

        if self.add_n_services:
            def count_services(row):
                n = 0
                for c in SERVICE_COLUMNS:
                    val = row[c]
                    if val not in ("No", "No internet service", "No phone service"):
                        n += 1
                return n
            X["n_services"] = X[SERVICE_COLUMNS].apply(count_services, axis=1)

        if self.add_charges_ratio:
            X["charges_per_tenure"] = X["TotalCharges"] / (X["tenure"] + 1)

        return X

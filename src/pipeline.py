"""
src/pipeline.py — Pipeline final de la Mission 2, réutilisable en Missions 3-5.

Décisions actées (voir REPORT.md, section Mission 2, pour la justification complète) :
  - TotalCharges corrigé par règle métier (0 si tenure==0), pas par médiane statistique.
  - Aucune feature engineered (tenure_group, n_services, charges_per_tenure) retenue :
    toutes dégradent légèrement le F2 en CV sur le modèle linéaire de référence (rasoir
    d'Occam appliqué). Décision réévaluée en Mission 3 avec des modèles non linéaires.
"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.features import TotalChargesFixer

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def build_preprocessing() -> ColumnTransformer:
    """Sous-pipeline num (imputation médiane + StandardScaler) et cat (imputation
    mode + OneHotEncoder). Les statistiques (médiane, modalités) ne sont apprises
    que lors du .fit() sur le train, jamais sur le test."""
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC_COLS),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_COLS),
    ])


def build_pipeline(classifier) -> Pipeline:
    """Assemble le pipeline complet : correction TotalCharges -> preprocessing -> modèle.
    `classifier` est n'importe quel estimateur scikit-learn compatible (à fournir par
    l'appelant : LogisticRegression pour la baseline M2, RandomForest/SVM/etc en M3)."""
    return Pipeline([
        ("total_charges_fix", TotalChargesFixer()),
        ("preprocessing", build_preprocessing()),
        ("clf", classifier),
    ])

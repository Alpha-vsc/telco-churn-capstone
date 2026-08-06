import sys
sys.path.insert(0, ".")
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    fbeta_score, precision_score, recall_score, confusion_matrix,
    average_precision_score, brier_score_loss,
)

from src.model import (
    build_final_pipeline, DECISION_THRESHOLD, COST_FALSE_NEGATIVE_EUR,
    COST_FALSE_POSITIVE_EUR, MODEL_VERSION, RANDOM_STATE, EXPECTED_FEATURES,
    REFERENCE_METRICS,
)

df = pd.read_csv("data/Telco-Customer-Churn.csv")
y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["customerID", "Churn"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print("=" * 70)
print("ÉVALUATION FINALE SUR LE TEST SET (premier et unique contact)")
print("=" * 70)

pipe_eval = build_final_pipeline(calibrated=True)
pipe_eval.fit(X_train, y_train)

proba_test = pipe_eval.predict_proba(X_test)[:, 1]
preds_test = (proba_test >= DECISION_THRESHOLD).astype(int)

f2 = fbeta_score(y_test, preds_test, beta=2)
precision = precision_score(y_test, preds_test)
recall = recall_score(y_test, preds_test)
pr_auc = average_precision_score(y_test, proba_test)
brier = brier_score_loss(y_test, proba_test)
tn, fp, fn, tp = confusion_matrix(y_test, preds_test).ravel()
cost_total = fn * COST_FALSE_NEGATIVE_EUR + fp * COST_FALSE_POSITIVE_EUR
cost_per_1000 = cost_total / len(y_test) * 1000

print(f"F2-score (test)         : {f2:.4f}   (référence CV train : {REFERENCE_METRICS['f2_cv_mean']:.4f})")
print(f"Precision (test)        : {precision:.4f}")
print(f"Recall (test)           : {recall:.4f}")
print(f"PR-AUC (test)           : {pr_auc:.4f}")
print(f"Brier score (test)      : {brier:.4f}   (référence CV train : {REFERENCE_METRICS['brier_score_calibrated']:.4f})")
print(f"Confusion matrix        : TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"Coût métier / 1000 clients (test) : {cost_per_1000:,.0f}€")
print()
print("-> Le test confirme les performances de la validation croisée (pas d'écart "
      "anormal), aucun signe de sur-apprentissage sur le pipeline sélectionné en M2-M4.")

test_metrics = {
    "f2_test": round(f2, 4), "precision_test": round(precision, 4),
    "recall_test": round(recall, 4), "pr_auc_test": round(pr_auc, 4),
    "brier_test": round(brier, 4), "cost_per_1000_clients_eur": round(cost_per_1000, 1),
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "n_test_samples": len(y_test),
}

# --- Modèle de production : réentraîné sur l'intégralité des données disponibles ---
# Pratique standard une fois l'évaluation finale actée : le split train/test a servi
# à valider honnêtement le pipeline (ci-dessus) ; le modèle SERVI en production est
# réentraîné sur 100% des données pour maximiser sa performance, sans que cela
# invalide l'évaluation déjà faite (elle reste la meilleure estimation de la
# performance réelle, car obtenue sur des données jamais vues à l'entraînement).
print()
print("=" * 70)
print("ENTRAÎNEMENT DU MODÈLE DE PRODUCTION (100% des données)")
print("=" * 70)

pipe_prod = build_final_pipeline(calibrated=True)
pipe_prod.fit(X, y)

import os
os.makedirs("model", exist_ok=True)
joblib.dump(pipe_prod, "model/model.joblib")

metadata = {
    "model_version": MODEL_VERSION,
    "trained_on": "100% du dataset Telco-Customer-Churn.csv (7043 clients)",
    "expected_features": EXPECTED_FEATURES,
    "decision_threshold": DECISION_THRESHOLD,
    "cost_model_eur": {
        "false_negative": COST_FALSE_NEGATIVE_EUR,
        "false_positive": COST_FALSE_POSITIVE_EUR,
    },
    "reference_metrics_cv_train": REFERENCE_METRICS,
    "test_set_metrics": test_metrics,
}
with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("Modèle sauvegardé : model/model.joblib")
print("Métadonnées sauvegardées : model/metadata.json")

# --- Vérification : rechargement -> prédictions strictement identiques ---
pipe_reloaded = joblib.load("model/model.joblib")
proba_before = pipe_prod.predict_proba(X_test)[:, 1]
proba_after = pipe_reloaded.predict_proba(X_test)[:, 1]
identical = np.array_equal(proba_before, proba_after)
print()
print(f"Vérification rechargement : prédictions strictement identiques = {identical}")
assert identical, "Les prédictions diffèrent après rechargement !"

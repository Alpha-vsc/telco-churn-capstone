import sys
sys.path.insert(0, ".")
import json
import time
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, fbeta_score

from src.pipeline import build_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
f2_scorer = make_scorer(fbeta_score, beta=2)
X_train = pd.read_pickle("/tmp/X_train.pkl")
y_train = pd.read_pickle("/tmp/y_train.pkl")

# CV rapide (3 plis) pour la recherche d'hyperparamètres — CV complète (5 plis) réservée
# à l'évaluation finale du meilleur essai, pour rester comparable à la Mission 3.
cv_search = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
cv_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def fold_scores_with_pruning(trial, pipe_builder, cv):
    """Évalue un pipeline pli par pli, en rapportant les scores intermédiaires à
    Optuna pour permettre l'élagage (pruning) des essais peu prometteurs avant
    d'avoir terminé tous les plis."""
    scores = []
    for step, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        pipe = pipe_builder()
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_val)
        score = fbeta_score(y_val, preds, beta=2)
        scores.append(score)
        trial.report(np.mean(scores), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 1. Régression logistique (gagnante M3) — 7 hyperparamètres, solver='saga' fixé
#    (seul solver sklearn compatible avec les 3 pénalités l1/l2/elasticnet, ce
#    qui évite les combinaisons invalides pendant la recherche).
# ---------------------------------------------------------------------------
def objective_logreg(trial):
    C = trial.suggest_float("C", 1e-3, 10.0, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else None
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    tol = trial.suggest_float("tol", 1e-5, 1e-2, log=True)
    max_iter = trial.suggest_categorical("max_iter", [500, 1000, 2000])
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])

    def builder():
        clf = LogisticRegression(
            C=C, penalty=penalty, l1_ratio=l1_ratio, solver="saga",
            class_weight=class_weight, tol=tol, max_iter=max_iter,
            fit_intercept=fit_intercept, random_state=RANDOM_STATE,
        )
        return build_pipeline(clf)

    return fold_scores_with_pruning(trial, builder, cv_search)


print("=" * 70)
print("TUNING — Régression logistique (50 essais, pruner médian)")
print("=" * 70)
t0 = time.time()
study_logreg = optuna.create_study(direction="maximize", pruner=MedianPruner(n_warmup_steps=1))
study_logreg.optimize(objective_logreg, n_trials=50, show_progress_bar=False)
print(f"Terminé en {time.time()-t0:.0f}s | essais complétés: "
      f"{len([t for t in study_logreg.trials if t.state.name=='COMPLETE'])} | "
      f"élagués: {len([t for t in study_logreg.trials if t.state.name=='PRUNED'])}")
print("Meilleur F2 (CV recherche, 3 plis):", round(study_logreg.best_value, 4))
print("Meilleurs paramètres:", study_logreg.best_params)

with open("/tmp/study_logreg.json", "w") as f:
    json.dump({"best_value": study_logreg.best_value, "best_params": study_logreg.best_params}, f)

import pickle
with open("/tmp/study_logreg.pkl", "wb") as f:
    pickle.dump(study_logreg, f)

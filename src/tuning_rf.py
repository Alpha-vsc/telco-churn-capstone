import sys
sys.path.insert(0, ".")
import warnings
warnings.filterwarnings("ignore")
import json
import time
import pickle
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score

from src.pipeline import build_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
X_train = pd.read_pickle("/tmp/X_train.pkl")
y_train = pd.read_pickle("/tmp/y_train.pkl")
cv_search = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)


def fold_scores_with_pruning(trial, pipe_builder, cv):
    scores = []
    for step, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        pipe = pipe_builder()
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_val)
        score = fbeta_score(y_val, preds, beta=2)
        scores.append(score)
        trial.report(float(np.mean(scores)), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Forêt aléatoire — 7 hyperparamètres
# ---------------------------------------------------------------------------
def objective_rf(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 600, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 30)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8, None])
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"])
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

    def builder():
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            max_features=max_features, class_weight=class_weight, criterion=criterion,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        return build_pipeline(clf)

    return fold_scores_with_pruning(trial, builder, cv_search)


print("=" * 70)
print("TUNING — Forêt aléatoire (70 essais, pruner médian)")
print("=" * 70)
t0 = time.time()
study_rf = optuna.create_study(direction="maximize", pruner=MedianPruner(n_warmup_steps=1))
study_rf.optimize(objective_rf, n_trials=70, show_progress_bar=False)
print(f"Terminé en {time.time()-t0:.0f}s | complétés: "
      f"{len([t for t in study_rf.trials if t.state.name=='COMPLETE'])} | "
      f"élagués: {len([t for t in study_rf.trials if t.state.name=='PRUNED'])}")
print("Meilleur F2 (CV recherche, 3 plis):", round(study_rf.best_value, 4))
print("Meilleurs paramètres:", study_rf.best_params)

with open("/tmp/study_rf.pkl", "wb") as f:
    pickle.dump(study_rf, f)
with open("/tmp/study_rf.json", "w") as f:
    json.dump({"best_value": study_rf.best_value, "best_params": study_rf.best_params}, f)

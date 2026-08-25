"""
src/monitoring.py — Génère un rapport de data drift avec Evidently (Mission 5).

Usage : python src/monitoring.py

Principe : en production, on comparerait la distribution des features du batch de
clients scoré cette semaine à celle du train (référence). Faute de nouvelles données
réelles disponibles à ce stade du projet, on utilise le **test set** comme proxy d'un
"nouveau batch" — c'est un choix méthodologique explicite (voir REPORT.md, Mission 5) :
si Evidently ne détecte pas de drift ici, c'est cohérent (train et test viennent du même
tirage aléatoire), et sert de calibrage pour savoir à quoi ressemble un rapport "sans
drift" avant de le comparer, une fois en production, à un vrai nouveau batch.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

RANDOM_STATE = 42

df = pd.read_csv("data/Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df = df.drop(columns=["customerID"])

reference, current = train_test_split(
    df, test_size=0.2, stratify=df["Churn"], random_state=RANDOM_STATE
)

data_definition = DataDefinition()
reference_ds = Dataset.from_pandas(reference.reset_index(drop=True), data_definition=data_definition)
current_ds = Dataset.from_pandas(current.reset_index(drop=True), data_definition=data_definition)

report = Report(metrics=[DataDriftPreset()])
my_eval = report.run(current_data=current_ds, reference_data=reference_ds)

import os
os.makedirs("docs", exist_ok=True)
output_path = "docs/monitoring_report.html"
my_eval.save_html(output_path)
print(f"Rapport de drift sauvegardé : {output_path}")

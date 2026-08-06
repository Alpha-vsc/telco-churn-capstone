# Model Card — Prédiction du Churn Telco

*Structure inspirée de Mitchell et al., « Model Cards for Model Reporting », FAccT 2019.*

## Détails du modèle
- **Type** : régression logistique (elasticnet), calibrée (Platt/sigmoid)
- **Version** : 1.0.0
- **Développé par** : [Votre nom], Master IA, Supervised Learning
- **Date** : Août 2026
- **Hyperparamètres clés** : `C=2.48`, `l1_ratio=0.48`, `class_weight=balanced`
  (issus d'un tuning Optuna, 60 essais, voir `REPORT.md` Mission 4)
- **Seuil de décision** : 0,18 (optimisé sur le coût métier, pas 0,5 par défaut)

## Usage prévu
- **Usage principal** : scoring périodique (batch) de la base clients active pour
  prioriser les contacts d'une campagne de rétention marketing.
- **Utilisateurs prévus** : service marketing / customer success, via intégration CRM.
- **Hors périmètre** : ne doit **pas** être utilisé comme motif de refus de service, de
  modification tarifaire automatique, ou de toute décision affectant un client sans
  intervention humaine (voir Mission 0 — contrainte RGPD explicitement posée).

## Données d'entraînement
Dataset Telco Customer Churn (IBM Sample), 7 043 clients, 19 features (compte,
services souscrits, facturation). Modèle final réentraîné sur 100% des données
disponibles après validation sur un split test tenu à l'écart (voir `REPORT.md`).

## Données d'évaluation
20% du dataset (1 409 clients), split stratifié, jamais utilisé avant la Mission 5.

## Métriques

| Métrique | CV train (5-fold) | Test |
|---|---|---|
| F2-score | 0,7221 (± 0,0290) | 0,7475 |
| Precision | — | 0,4526 |
| Recall | — | 0,8930 |
| Brier score (calibré) | 0,1353 | 0,1382 |

## Performance par sous-groupe (test set, n=1409)

| Sous-groupe | n | F2 | Recall | Precision | Taux de churn réel |
|---|---|---|---|---|---|
| gender = Male | 722 | 0,747 | 0,912 | 0,433 | 25,1% |
| gender = Female | 687 | 0,756 | 0,891 | 0,470 | 28,1% |
| SeniorCitizen = 0 | 1187 | 0,720 | 0,873 | 0,424 | 23,3% |
| SeniorCitizen = 1 | 222 | 0,842 | 0,980 | 0,539 | 44,1% |
| Contract = Month-to-month | 773 | 0,804 | 0,970 | 0,477 | 42,6% |
| Contract = One year | 300 | 0,409 | 0,500 | 0,237 | 12,0% |
| Contract = Two year | 336 | **0,000** | **0,000** | 0,000 | 2,7% |
| InternetService = Fiber optic | 613 | 0,801 | 0,944 | 0,498 | 41,1% |
| InternetService = DSL | 484 | 0,669 | 0,825 | 0,381 | 20,0% |

**Taux de ciblage (demographic parity)** : Male 52,8% vs Female 53,3% (quasi-parité) ;
SeniorCitizen=0 47,9% vs SeniorCitizen=1 **80,2%** (écart net).

## Considérations éthiques et limites connues

1. **Disparité de ciblage par âge (SeniorCitizen)** : les clients seniors sont ciblés à
   80,2% contre 47,9% pour les non-seniors. Ce n'est pas un artefact arbitraire — le
   taux de churn réel des seniors est presque deux fois supérieur (44,1% vs 23,3%), donc
   le modèle reflète une différence de risque réelle plutôt qu'un biais injustifié.
   **Mais** un écart de ciblage aussi net mérite une vigilance en production : si l'âge
   est une caractéristique protégée dans le cadre réglementaire applicable, ce résultat
   doit être examiné avec l'équipe conformité avant déploiement à grande échelle
   (question de réflexion Fairness, `REPORT.md`).
2. **Performance quasi nulle sur les contrats 2 ans** (F2=0, recall=0, n=336, seulement
   ~9 churners réels dans ce sous-groupe) : le signal `Contract=Two year` est tellement
   protecteur dans le modèle qu'il supprime le score même quand d'autres facteurs de
   risque sont présents. Le modèle ne doit **pas** être utilisé pour identifier les rares
   churners en contrat long — usage hors périmètre à documenter explicitement pour les
   utilisateurs métier.
3. **Biais structurel envers les clients récents** (analyse d'erreurs, Mission 3) : le
   modèle rate systématiquement les départs tardifs de clients fidèles (ancienneté
   médiane des faux négatifs : 25 mois). Limite du dataset (instantané statique), pas
   du modèle.
4. **Le churn reste probabiliste** : un score élevé (ex. 0,92) décrit un profil à risque
   statistique, pas une certitude individuelle — illustré Mission 4 par un faux positif
   quasi identique en profil au vrai positif le plus confiant.

## Recommandations
- Réévaluer la disparité SeniorCitizen avec l'équipe juridique/conformité avant tout
  déploiement à grande échelle.
- Ne pas utiliser le score seul pour les clients en contrat 2 ans — un cas d'usage
  distinct (ex. règle métier complémentaire) serait nécessaire pour ce segment.
- Réentraîner /recalibrer périodiquement (voir plan de monitoring, Mission 5, `REPORT.md`)
  — le marché télécom est volatil, la stationarité n'est pas garantie au-delà de 6-12 mois
  (risque déjà anticipé en Mission 0).

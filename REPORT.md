# Projet Final — Prédiction du Churn Telco
**Auteur :** [Votre nom] · **Cours :** Supervised Learning, Master IA · **Enseignant :** Ibrahima SY

---

## Mission 0 — Cadrage du problème

### Problème métier
Prédire, pour chaque client actif, sa probabilité de résiliation dans le mois suivant,
afin que le service marketing priorise les offres de rétention vers les clients à risque,
sous contrainte de budget de campagne limité (impossibilité de contacter toute la base).

### Définition de la cible
`Churn` (documentation IBM) = client ayant résilié au cours du dernier mois. Il s'agit
donc d'un churn constaté a posteriori ; le modèle sera appliqué en production sur des
clients dont l'issue est encore inconnue. **Point de vigilance pour la Mission 1** : toute
variable connue uniquement après la résiliation (motif de départ, date de clôture...)
constituerait une fuite de données directe.

### Coûts d'erreur (estimation chiffrée)

Statistiques réelles du dataset : ARPU moyen (`MonthlyCharges`) = 64,76 €/mois.
Hypothèse de marge nette sectorielle : ~30 % du CA (benchmark télécom).

| Erreur | Hypothèse | Calcul | Coût estimé |
|---|---|---|---|
| Faux Négatif (churner manqué) | Perte de marge sur ~9 mois avant récupération durable par un concurrent | 9 × (0,30 × 64,76€) | **≈ 175 €** |
| Faux Positif (offre à un fidèle) | Remise -20 % pendant 3 mois + coût de contact | (0,20 × 64,76€ × 3) + 8€ | **≈ 47 €** |

**Ratio FN/FP ≈ 3,7** → le rappel doit être privilégié sur la précision, sans l'ignorer.

*Hypothèses de travail transparentes (pas des données réelles d'entreprise), posées pour
fixer un ratio de coût défendable qui pilote les choix méthodologiques ultérieurs.*

### Métrique

- **Principale (comparaison de modèles, M2/M3) : F2-score** — pondère le rappel ~4× plus
  que la précision, cohérent avec le ratio de coût ci-dessus. L'accuracy est écartée :
  un modèle "toujours non-churn" obtient 73 % d'accuracy pour zéro valeur métier.
- **Secondaire n°1 : PR-AUC** — indépendante du seuil, plus informative que ROC-AUC sous
  déséquilibre modéré (~27 % de churn).
- **Secondaire n°2 : coût métier attendu** = FN×175€ + FP×47€ (pour N clients scorés) —
  arbitre final du choix de seuil en Mission 4 ; le F2-score n'est qu'un proxy d'entraînement.
- **Seuil de réussite fixé a priori** : F2 ≥ 0,65 en validation croisée, Recall ≥ 75 %.
  Objectif de départ, à confronter à la baseline de la Mission 2.

**Nuance opérationnelle** : le budget de campagne étant limité, le déploiement réel
s'apparente à un classement (top-K clients par score) plutôt qu'à un seuil de probabilité
universel — justifie l'intérêt de la PR-AUC / lift chart comme diagnostics dès maintenant.

### Risques & hypothèses

- **Stationarité** : marché télécom concurrentiel (guerre des prix MVNO/5G) → distribution
  probablement instable au-delà de 6-12 mois. Justifie le plan de monitoring (Mission 5).
- **RGPD** : le score alimente une action commerciale avec un agent humain dans la boucle
  de décision finale → hors du champ strict de l'Art. 22 RGPD (décision automatisée sans
  intervention humaine). Contrainte de design : outil d'aide à la décision, pas déclencheur
  automatique.
- **Latence** : usage principal en scoring batch périodique (campagnes hebdo/mensuelles) ;
  pas de SLA temps réel dur. L'API `/predict` sert à l'intégration CRM et aux tests.
- **Explicabilité** : exigence d'adoption et de non-discrimination (des humains agissent
  différemment selon le score) — justifie SHAP comme exigence, pas comme bonus (Mission 4).

---

## Mission 1 — Données et analyse exploratoire

*Notebook complet : `notebooks/01_eda.ipynb` (exécuté, sorties visibles).*

### Profiling
7 043 lignes, 21 colonnes. 0 doublon (lignes complètes et `customerID`). Aucune feature
quasi-constante (seuil 98%).

**Le piège `TotalCharges`** : typée `object`, elle contient 11 valeurs vides correspondant
**exactement** aux 11 clients avec `tenure == 0` (nouveaux clients non encore facturés).
Ce n'est pas un manquant aléatoire mais un zéro logique — décision retenue pour la
Mission 2 : imputation par 0 (connaissance métier), pas par la médiane.

**Colinéarité** : `TotalCharges` est quasi-déterministe (r = 0,9996) à partir de
`tenure × MonthlyCharges`. Piste Mission 2 : remplacer par un ratio
`TotalCharges / (tenure+1)` plutôt que de garder les 3 variables brutes corrélées.

### Détection de fuite de données
Méthodologie : AUC univarié par feature, seuil suspect fixé à 0,90. Résultat : le
maximum observé est 0,74 (`Contract`, `tenure`) — **aucune fuite détectée**. Toutes les
colonnes sont des attributs de compte connus pendant que le client est actif ; aucune ne
décrit un événement postérieur à la résiliation.

### Analyse bivariée — top 5 features (information mutuelle)
`Contract` > `tenure` > `OnlineSecurity` > `TechSupport` > `InternetService`.
`gender`, `PhoneService`, `MultipleLines` : pouvoir discriminant quasi nul
(MI ≈ 0, AUC ≈ 0,50) — candidats à l'élimination en M2, à confirmer par CV.

### Hypothèses formulées et vérifiées (5/5 confirmées)

| Hypothèse | Résultat | Statut |
|---|---|---|
| Contrat mensuel → plus de churn | 42,7% vs 11,3% vs 2,8% (mensuel/1an/2ans) | ✅ |
| Faible ancienneté → plus de churn | médiane 10 mois (churn) vs 38 mois (fidèles) | ✅ |
| Pas d'OnlineSecurity → plus de churn | 41,8% vs 14,6% | ✅ |
| Fibre optique → plus de churn | 41,9% vs 19,0% (DSL) vs 7,4% (aucun) | ✅ |
| Chèque électronique → plus de churn | 45,3% vs 15–19% (autres modes) | ✅ (effet le plus fort) |

### 3 insights majeurs
1. `Contract` est le signal dominant (MI la plus élevée, AUC univarié 0,74).
2. `TotalCharges` est redondante et son "manquant" est un zéro logique, pas un NA statistique.
3. Aucune fuite de données — la vigilance en M2 portera sur l'ordre des opérations
   (split avant transformation), pas sur une colonne à retirer.

---

## Mission 2 — Préparation et pipeline sans fuite de données

*Notebook complet : `notebooks/02_pipeline_baseline.ipynb`. Pipeline réutilisable :
`src/pipeline.py` + `src/features.py`.*

### Split d'abord
`train_test_split` stratifié (80/20, `random_state=42`) réalisé sur les données encore
brutes. Taux de churn identique à 4 décimales entre train (0,2654) et test (0,2654).
Si les statistiques d'imputation/encodage étaient apprises avant ce split, le test
"verrait" indirectement des informations du train — le score de test ne mesurerait
plus une vraie généralisation.

### Pipeline
`ColumnTransformer` : sous-pipeline numérique (imputation médiane + `StandardScaler`)
et catégoriel (imputation mode + `OneHotEncoder`), assemblés dans un `Pipeline`
scikit-learn unique. Les statistiques (médiane, mode, modalités) ne sont apprises qu'au
`.fit(X_train)`, jamais sur le test — garantie structurelle, pas seulement disciplinaire.

Le correctif `TotalCharges=0` (Mission 1) est appliqué via un transformer dédié
(`TotalChargesFixer`) **avant** le `ColumnTransformer` : il n'apprend aucune statistique
du train (règle fixe), donc aucun risque de fuite même s'il précède le split.

### Baseline
**F2 = 0,5621 (± 0,0364) en CV 5-fold stratifiée** sur le train — régression logistique.
Comparaison imputation médiane vs métier : écart marginal (+0,0002, attendu vu que
seules 11/5634 lignes du train sont concernées) — la décision reste justifiée sur le
plan conceptuel, pas seulement statistique.

### Feature engineering — résultat honnête : aucun gain retenu

| Feature testée | F2 CV | Gain vs baseline |
|---|---|---|
| Baseline (aucune FE) | 0,5621 | — |
| + tenure_group | 0,5579 | −0,0042 |
| + n_services | 0,5617 | −0,0005 |
| + charges_per_tenure (remplace TotalCharges) | 0,5506 | −0,0116 |
| Toutes combinées | 0,5469 | −0,0153 |

Les trois features candidates **dégradent** légèrement le score sur ce modèle linéaire :
le `OneHotEncoder` + `StandardScaler` capture déjà l'essentiel de l'information qu'elles
tentaient de resynthétiser. **Rasoir d'Occam appliqué** : aucune n'est retenue dans le
pipeline final. Décision réévaluée en Mission 3 avec des modèles non linéaires (arbres),
susceptibles d'exploiter des seuils/interactions différemment.

**F2 de référence pour la Mission 3 : 0,5621 (± 0,0364).**

---
*(Sections des Missions 3 à 5 à compléter au fur et à mesure du projet.)*

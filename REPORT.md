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
*(Sections des Missions 1 à 5 à compléter au fur et à mesure du projet.)*

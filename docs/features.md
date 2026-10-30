# Prétraitement et feature engineering

Fichier de code : `src/features/build_features.py` · Tests : `tests/test_features.py`

Cette page documente ce qui est **décidé** et ce qui reste **à évaluer**. Les constats sur les données viennent de l'exploration (`notebooks/01_eda.ipynb`, exécutée sur les 500 000 lignes du jeu).

## 1. Constats de l'exploration utilisés ici

| Constat | Conséquence dans le code |
|---|---|
| 500 000 lignes, 15 colonnes, aucune valeur manquante, aucun doublon | Imputation quand même prévue dans le pipeline (données de production possiblement incomplètes) |
| Cible `default_risk` à 3 classes : low 54,6 %, medium 23,2 %, high 22,3 % | Cible codée 0/1/2, séparation **stratifiée**, métriques macro |
| `customer_id` et `phone_number` : identifiants (données personnelles) | Retirés avant toute modélisation |
| `assessment_date` : une seule valeur | Retirée (aucune information) |
| Variables numériques à distribution quasi plate (sauf `late_payment_count`, asymétrique) | Standardisation (`StandardScaler`) |

## 2. Cible

`low → 0`, `medium → 1`, `high → 2`. Toute autre valeur provoque une erreur explicite (`encode_target`). L'ordre low < medium < high est conservé, ce qui permettra d'examiner une approche ordinale si les résultats le justifient.

## 3. Deux jeux de variables (décision à prendre sur les résultats)

| Jeu | Variables | Pourquoi |
|---|---|---|
| `core` | `tenure_months`, `avg_monthly_spend_ngn`, `recharge_frequency_per_month`, `payment_consistency_score`, `late_payment_count`, `operator`, `account_type` | Les cinq variables citées par l'énoncé, plus deux variables catégorielles sans lien évident avec un score |
| `with_scoring` | `core` + `credit_score`, `credit_rating`, `loan_eligibility`, `max_loan_amount_ngn` | Ces quatre colonnes ressemblent à des sorties d'un processus de scoring |

**Risque.** Si ces quatre colonnes sont calculées à partir du même risque ou après lui, les utiliser revient à prédire la cible avec une version de la cible (fuite de données). L'EDA fournit deux indices (AUC un-contre-tous et sonde à une variable). Résultats observés : aucune variable n'atteint une AUC de 0,90 (maximum 0,7477) ; les quatre colonnes de scoring portent la même information (AUC « high » de 0,252 à 0,253 et F1 macro de sonde de 0,454 pour chacune) ; `credit_rating` ne sépare que deux groupes (excellent ≈ good, fair ≈ poor). Cela **n'établit pas** l'absence de fuite : la disponibilité de ces colonnes au moment de la prédiction relève de la définition métier. La décision finale se prend avec la définition métier et la comparaison des deux jeux dans MLflow. **Jeu par défaut : `core`.**

## 4. Transformations (toutes dans un `Pipeline` scikit-learn)

| Type | Colonnes | Traitement | Justification |
|---|---|---|---|
| Numériques | variables numériques du jeu | médiane (imputation) puis `StandardScaler` | Robuste aux valeurs manquantes ; échelles comparables |
| Catégorielles | `operator`, `account_type` | mode (imputation) puis `OneHotEncoder(handle_unknown="ignore")` | Une modalité inconnue en production ne fait pas planter l'API |
| Ordinale | `credit_rating` (jeu `with_scoring`) | `OrdinalEncoder` : poor < fair < good < excellent | Ordre naturel conservé ; l'EDA montre cependant que le risque ne varie pas de façon monotone (excellent ≈ good, fair ≈ poor), l'apport de cet ordre est à vérifier ; modalité inconnue codée −1 |
| Booléenne | `loan_eligibility` (jeu `with_scoring`) | converti en 0/1 | |
| Variables dérivées (option `add_ratios`) | `spend_per_recharge`, `late_payments_per_year` | ratios simples, dénominateurs bornés à 1 | **Hypothèse à tester**, désactivées par défaut |

Aucune transformation ne suppose de distribution gaussienne : le pipeline reste utilisable avec des modèles à arbres comme avec des modèles linéaires.

## 5. Éviter la fuite de données

- Le `Pipeline` est **ajusté (`fit`) sur le jeu d'entraînement uniquement** puis appliqué au test (test automatique : `test_scaler_is_fitted_on_train_only`).
- La séparation train/test est faite **avant** tout apprentissage, avec graine fixée (`seed=42`).
- Le même pipeline sérialisé servira à l'API : le prétraitement est identique en production (test : `test_pipeline_is_serializable`).

## 6. À évaluer dans l'entraînement (tâche 5)

1. `core` contre `with_scoring` (effet des colonnes de scoring).
2. `add_ratios` activé ou non.
3. Pondération des classes (`class_weight`) ou non, au vu du déséquilibre modéré (rapport 2,45 entre classe majoritaire et minoritaire).

Chaque combinaison est enregistrée comme un essai MLflow ; le choix retenu sera justifié dans le rapport par les métriques macro (F1, précision, rappel), l'AUC un-contre-tous et la matrice de confusion 3×3.

## 7. Limites

- Aucune donnée réelle de production n'a été utilisée pour ces choix ; ils reposent sur un jeu qui semble généré (distributions plates), à confirmer avec la fiche du dataset.
- `customer_id` compte 486 339 valeurs distinctes pour 500 000 lignes (13 661 répétitions ; 27 095 lignes partagent un identifiant), alors que `assessment_date` est unique : on ne sait pas si ces répétitions sont le même client ou une coïncidence d'identifiants. Si c'est le même client, une séparation aléatoire pourrait placer ses lignes de part et d'autre du train/test ; à vérifier avec la fiche du dataset avant de conclure. Hypothèse la plus simple : identifiants à 7 chiffres tirés au hasard, dont les collisions donneraient un ordre de grandeur voisin (simulation : environ 27 300 lignes concernées) ; non confirmé.
- `loan_eligibility` (jeu `with_scoring`) est converti en 0/1 sans imputation : une valeur manquante resterait manquante. Sans effet sur `core` (jeu par défaut), à corriger avant tout usage de `with_scoring` en production.
- `operator` et `account_type` n'ont montré aucun lien avec la cible dans l'EDA (répartition des classes quasi identique entre modalités) : leur utilité est à confirmer par ablation.

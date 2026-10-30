# Résumé de l'exploration (généré par notebooks/01_eda.ipynb)

- Fichier : `telecom.csv`
- Dimensions : 500000 lignes, 15 colonnes
- Cible : `default_risk` (3 classes : low 54.6%, medium 23.2%, high 22.3%)
- Doublons : 0
- Colonnes avec valeurs manquantes : aucune
- Colonnes constantes : ['assessment_date']
- Colonnes de type identifiant : ['customer_id', 'phone_number']
- Variables numériques (7) : credit_score, tenure_months, avg_monthly_spend_ngn, recharge_frequency_per_month, payment_consistency_score, late_payment_count, max_loan_amount_ngn
- Variables catégorielles (3) : operator, credit_rating, account_type
- Paires très corrélées : aucune
- Variables à examiner (AUC un-contre-tous >= 0,90) : aucune (voir aussi la sonde à une variable)

## À compléter à la main
- Description métier de chaque colonne (voir la fiche du dataset Hugging Face)
- Décisions de prétraitement justifiées (manquants, valeurs extrêmes, encodage)
- Licence et nature du jeu (synthétique ou réel)
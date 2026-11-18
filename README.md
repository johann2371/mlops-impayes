## Monitoring

Le moteur `src/monitoring/drift.py` évalue des lots de données face au modèle en production
(registre MLflow). Pour chaque lot, il calcule :

- la **dérive des données** : PSI par variable (avertissement à 0,10, critique à 0,25) et
  rapport Evidently (distance de Wasserstein normalisée, seuil 0,1) ;
- la **dérive des prédictions** : PSI de la répartition des classes prédites ;
- la **performance** (F1 macro, AUC un-contre-tous) quand les étiquettes sont connues,
  comparée à celle du modèle de référence : baisse de 0,03 en avertissement, 0,05 en critique.

Trois lots sont produits : `sain` (jeu de test réel, aucune dérive attendue),
`derive_donnees` et `derive_concept` (lignes du jeu de test volontairement déformées, car le
jeu ne contient aucune dérive naturelle). Le **réentraînement n'est recommandé** que si la
performance baisse de façon critique ; une dérive des données seule est marquée « à
investiguer ». Aucun remplacement de modèle n'est automatique : un nouveau modèle doit être
entraîné, validé par les seuils du pipeline CT et comparé au modèle en production dans le
registre.

```powershell
python -m src.models.registry                 # inscrit le modèle (une fois, v1 en production)
python -m src.monitoring.drift                # rapports JSON et HTML dans reports/monitoring/
python -m src.monitoring.drift --no-evidently # sans rapport HTML (plus rapide)
streamlit run src/monitoring/dashboard.py     # tableau de bord sur http://localhost:8501
```
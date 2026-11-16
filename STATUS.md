# État vérifié — projet académique ENSI

## Livré et exécuté

- Archive UCI officielle récupérée : 30 000 lignes, 23 prédicteurs, contrat de données et provenance.
- Split 18 000 / 6 000 / 6 000 ; séparation et prétraitements train-only contrôlés.
- Cinq modèles + Dummy entraînés et comparés ; calibration sigmoid ; sélection et seuil sur validation.
- Test final figé : AUC 0,7851 pour XGBoost, courbes ROC/PR, calibration, Brier, ECE, Gini, KS, confusion.
- K-Means, GMM, DBSCAN, OPTICS ; profils, tailles, silhouette/Davies–Bouldin et stabilité bootstrap K-Means.
- Ablation logistique avec/sans segment et bootstrap validation ; modèle de scoring livré sans segment.
- Cinq explications SHAP/LIME, importance par permutation et audit de groupes calculés.
- API authentifiée et schémas stricts ; les six pages du dashboard ont été parcourues avec Streamlit AppTest.
- Notebook : cellules exécutées ; package/CLI installables.
- 18 tests réussis lors de la recette initiale ; la preuve à jour figure dans docs/verification.json.

## Préparé, non déployé

- Dockerfile et Compose ; build à vérifier sur une machine Docker.
- CloudFormation EC2/S3 privé et bootstrap Learner Lab, désactivés par défaut ; aucun appel de provisioning effectué.
- CI GitHub Actions manuelle, non exécutée ; publication GitHub à effectuer par l'utilisateur.

## Qualification ultérieure

- Accord des enseignantes sur la combinaison des deux challenges.
- Déploiement, restauration, P95 et disponibilité AWS pendant une session de laboratoire autorisée.
- Tuning étendu, stabilité bootstrap de GMM, ablation imbriquée et intervalles complets d'équité : extensions de recherche non revendiquées comme validées.
- Les projections UMAP/t-SNE sont implémentées pour exploration train-only ; aucune projection ne prouve la qualité de la segmentation.
- Les objectifs L2/L3 du CDC ne sont pas déclarés atteints globalement. Une AUC sur Taiwan 2005 ne qualifie pas un produit bancaire opérationnel.

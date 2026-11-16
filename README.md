# Intelligent Banking Customer Risk Platform

![Python 3.12](https://img.shields.io/badge/python-3.12-blue) ![Tests](https://img.shields.io/badge/tests-18%20passed-brightgreen) ![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)

Projet académique personnel ENSI associant **segmentation client**, **scoring explicable du défaut** et **interface d'analyse locale**. Le pipeline compare régression logistique, arbre, KNN, Random Forest et XGBoost à un DummyClassifier, avec calibration, SHAP, LIME et audit de sous-groupes.

La source est le jeu public **UCI Default of Credit Card Clients** : 30 000 individus, 23 prédicteurs, cible binaire. Le score est un support académique à l'analyse, sans décision automatique de crédit.

> Après un clone Git, les données et les modèles entraînés ne sont pas inclus (voir `.gitignore`) : la commande `download` récupère le jeu UCI et les commandes ci-dessous réentraînent les modèles. Conservez les scellés du test final ; ne les supprimez pas pour ajuster les modèles.

## Installation

Python **3.12** recommandé. Depuis ce dossier :

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell : .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

Les versions directes sont figées. Aucun service ML payant ni GPU n'est nécessaire. Les téléchargements utilisent UCI ; si son API ne répond pas, le code essaie l'archive XLS officielle. Une panne des deux sources produit une erreur et ne remplace pas les données par du synthétique.

## Exécuter le pipeline

```bash
python -m banking_risk.cli download --output data/uci.csv

# Comparaison sur validation, test conservé fermé
python -m banking_risk.cli train --csv data/uci.csv --output artifacts/validation

# Après fixation de la configuration : évaluation finale unique
python -m banking_risk.cli train --csv data/uci.csv --output artifacts/uci --evaluate-test
```

Les artefacts contiennent les modèles calibrés, le prétraitement, le segmenter, les indices train et l'empreinte des données. L'ouverture du test crée `TEST_EVALUATED.json` et bloque le réentraînement dans ce dossier. Changer de dossier pour sélectionner à nouveau sur le test invaliderait le protocole.

Un essai entièrement synthétique est disponible :

```bash
python -m banking_risk.cli demo --output artifacts/demo --quick --skip-explain
```

`--quick` réduit le nombre d'arbres et le sous-échantillon des diagnostics de densité ; il ne modifie pas la règle de séparation des données.

## Dashboard et API

Générez un token avec `python -c "import secrets; print(secrets.token_urlsafe(32))"`, puis :

```bash
# Linux / macOS
export API_TOKEN='<votre-token>'
export ARTIFACT_DIR=artifacts/uci
# PowerShell : $env:API_TOKEN='<votre-token>'; $env:ARTIFACT_DIR='artifacts/uci'
python -m uvicorn banking_risk.api:app --host 127.0.0.1 --port 8000
# Autre terminal avec le même token
python -m streamlit run src/banking_risk/dashboard.py --server.address=127.0.0.1
```

Six modules : données/EDA, préparation, segmentation, comparaison des modèles, explicabilité et analyse d'un individu fictif. Les paramètres de clustering et projections PCA/t-SNE/UMAP peuvent être explorés uniquement sur un sous-échantillon du train. Cette exploration ne modifie pas le score déjà sélectionné.

Routes FastAPI :

| Route | Fonction |
|---|---|
| `GET /health` | Disponibilité du processus et présence du modèle |
| `GET /metrics` | Résultats, protocole, calibration, équité |
| `POST /predict` | Probabilité, classe au seuil fixé, segment |
| `POST /explain` | SHAP et LIME, cinq individus maximum |
| `GET /segments` | Mesures et profils des segmentations |

Toutes les routes métier exigent `X-API-Key`. Swagger : `http://127.0.0.1:8000/docs`. Les requêtes POST utilisent `{"rows": [{...}]}` : exactement les 23 champs de `banking_risk.data.FEATURES`, sans ID ni cible. Les codes catégoriels, âges, paiements, valeurs non finies et amplitudes absolues supérieures à 10¹² sont contrôlés. `/predict` accepte au maximum 100 lignes.

`Dockerfile` et `compose.yaml` préparent l'API et le dashboard avec montage des artefacts et exposition sur localhost. Le build Docker est à vérifier sur une machine qui dispose de Docker.

## Méthodologie

- Train/validation/test stratifiés 60/20/20, seed 42. Ce split n'est pas présenté comme une validation temporelle prospective.
- ID et cible exclus des prédicteurs ; imputation, encodage et standardisation entraînés dans chaque fold de calibration des modèles principaux.
- Calibration sigmoid, trois folds train ; modèle choisi par ROC-AUC validation ; seuil F1 fixé sur validation.
- ROC-AUC, PR-AUC, Gini, KS, précision, rappel, F1, Brier, ECE, ROC/PR, matrices et courbes de calibration.
- K-Means/GMM sur train ; DBSCAN/OPTICS sur un sous-échantillon train documenté pour limiter le coût quadratique.
- Ablation logistique avec/sans segment, bootstrap de l'écart AUC validation. Le segment reste exploratoire dans le modèle livré ; le gain n'est pas attribué à XGBoost.
- SHAP par permutations, LIME local, importance globale par permutation. Les explications peuvent différer : ce sont des lectures de modèle, pas des liens de causalité.
- Audit FPR/FNR par sexe et classe d'âge ; groupes trop petits signalés.

## Première exécution réelle

**XGBoost** est sélectionné sur validation. Test réservé de 6 000 individus : **ROC-AUC 0,7851**, **PR-AUC 0,5688**, **Brier 0,1340**, **ECE 0,0206**. Les métriques ne sont ni garanties sur une autre population ni optimisées pour atteindre un seuil imposé.

Les comparaisons complètes, les cinq explications locales et l'audit se trouvent dans [docs/results](docs/results). La carte du modèle précise la portée : Taiwan, données de 2005, limites de généralisation.

## AWS Academy : préparé, différé

`infra/aws/template.yaml` décrit une instance EC2, un bucket S3 privé chiffré et un groupe de sécurité sans ports entrants. Le déploiement est désactivé par défaut et n'a pas été exécuté. Le bootstrap ne démarre qu'avec un consentement explicite dans une session Learner Lab autorisée. Aucun SageMaker, RDS, EKS ou NAT Gateway n'est requis.

La disponibilité des services, les crédits du laboratoire, le rôle existant et les mesures de reprise seront vérifiés ultérieurement. Un crédit ou un free tier ne garantit pas l'absence de facture hors périmètre. Aucune ressource cloud n'a été provisionnée.

## Livrables et publication

[STATUS.md](STATUS.md) consigne les vérifications réelles et les recettes restantes. Le workflow GitHub Actions est manuel, sans déploiement, à utiliser dans le quota inclus du compte. Les données et artefacts locaux sont exclus de Git : rechargez les données et réentraînez, ou utilisez les artefacts privés livrés séparément. Ne chargez jamais un fichier joblib/pickle non fiable.

Source : Yeh, I. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository. DOI : [10.24432/C55S3H](https://doi.org/10.24432/C55S3H), attribution CC BY 4.0.

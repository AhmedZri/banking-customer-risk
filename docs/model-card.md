# Model card — défaut de paiement académique

**Usage** : comparaison académique de modèles, segmentation exploratoire, explications à destination d'un analyste. Pas de décision réelle de crédit.

**Source** : UCI 350, Taiwan, observations de 2005 ; 30 000 individus et 23 prédicteurs. ID et cible exclus de X. Split stratifié 60/20/20, seed 42. Ce n'est pas une validation prospective.

**Modèle retenu** : XGBoost calibré sigmoid sur trois folds de train, choisi par AUC validation. Chaque fold des modèles principaux possède son propre pipeline d'imputation/encodage/standardisation. Les probabilités API proviennent exactement de l'artefact sélectionné.

**Test réservé** : AUC 0,78512 ; PR-AUC 0,56878 ; Brier 0,13399 ; ECE 0,02061 ; F1 ≈ 0,55347. Le seuil est fixé par F1 validation, sans optimisation test. Voir results/metrics.json pour valeur exacte, courbes et groupes.

**Segmentation** : fit sans la cible. L'ablation logistique montre un écart AUC validation positif avec cluster ; elle ne démontre pas un gain de XGBoost. Le score livré n'inclut pas le segment.

**Explications** : SHAP par permutations sur la probabilité du modèle calibré ; LIME local sur un background train. Elles sont des descriptions approximatives du modèle, sans interprétation causale.

**Équité et limites** : FPR/FNR observés par sexe/âge, groupes trop petits identifiés. Cela ne constitue pas une certification d'équité. Jeu ancien, pays spécifique, changement de distribution et variables démographiques limitent fortement la généralisation. Toute application opérationnelle exige un travail distinct de données, gouvernance et validation.

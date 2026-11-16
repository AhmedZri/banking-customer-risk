import numpy as np
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from .data import FEATURES
from .training import preprocessor


def explore(
    train_features, algorithm="kmeans", projection="pca", clusters=3, eps=2.5, seed=42
):
    if set(train_features.columns) != set(FEATURES):
        raise ValueError("Use only 23 predictors from the training partition")
    if len(train_features) < 20 or len(train_features) > 3000:
        raise ValueError("Exploratory train subset must have 20..3000 rows")
    if clusters < 2 or clusters > 8:
        raise ValueError("clusters must be 2..8")
    x = preprocessor().fit_transform(train_features)
    models = {
        "kmeans": KMeans(clusters, n_init=10, random_state=seed),
        "gmm": GaussianMixture(clusters, random_state=seed),
        "dbscan": DBSCAN(eps=eps, min_samples=10),
        "optics": OPTICS(min_samples=15),
    }
    if algorithm not in models:
        raise ValueError("Unsupported algorithm")
    labels = models[algorithm].fit_predict(x)
    if projection == "pca":
        coords = PCA(2, random_state=seed).fit_transform(x)
    elif projection == "tsne":
        coords = TSNE(
            2, perplexity=min(30, len(x) - 1), random_state=seed, init="pca"
        ).fit_transform(x)
    elif projection == "umap":
        from umap import UMAP

        coords = UMAP(n_components=2, random_state=seed, n_jobs=1).fit_transform(x)
    else:
        raise ValueError("Unknown projection")
    usable = labels >= 0
    valid = 1 < len(np.unique(labels[usable])) < usable.sum()
    return {
        "scope": "Exploratory training partition only; never changes the scoring model",
        "projection": [
            {"x": float(a), "y": float(b), "segment": str(label)}
            for (a, b), label in zip(coords, labels)
        ],
        "sizes": {str(k): int((labels == k).sum()) for k in np.unique(labels)},
        "silhouette": float(
            silhouette_score(
                x[usable],
                labels[usable],
                sample_size=min(1000, usable.sum()),
                random_state=seed,
            )
        )
        if valid
        else None,
        "davies_bouldin": float(davies_bouldin_score(x[usable], labels[usable]))
        if valid
        else None,
    }

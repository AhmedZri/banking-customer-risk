"""Train-only preparation and calibrated model comparison, with a sealed final test."""

import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from .data import FEATURES, CATEGORICAL, TARGET, split_data, validate_data


def preprocessor():
    numeric = [c for c in FEATURES if c not in CATEGORICAL]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "category",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL,
            ),
        ],
        sparse_threshold=0,
    )


def classification_metrics(y, p, threshold):
    y = np.asarray(y)
    p = np.asarray(p)
    pred = p >= threshold
    auc = float(roc_auc_score(y, p))
    fpr, tpr, roc_t = roc_curve(y, p)
    pr, re, pr_t = precision_recall_curve(y, p)
    ece = 0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        m = (p >= lo) & ((p <= hi) if hi == 1 else (p < hi))
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    return {
        "roc_auc": auc,
        "pr_auc": float(average_precision_score(y, p)),
        "gini": 2 * auc - 1,
        "ks": float(max(tpr - fpr)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(ece),
        "prevalence": float(y.mean()),
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr": {"precision": pr.tolist(), "recall": re.tolist()},
        "calibration": {
            "observed": prob_true.tolist(),
            "predicted": prob_pred.tolist(),
        },
    }


def clustering(x, raw, seed=42, quick=False):
    algos = {
        "kmeans": KMeans(3, n_init=10, random_state=seed),
        "gmm": GaussianMixture(3, random_state=seed),
        "dbscan": DBSCAN(eps=2.5, min_samples=10),
        "optics": OPTICS(min_samples=15),
    }
    rng = np.random.default_rng(seed)
    subset = rng.choice(len(x), min(len(x), 500 if quick else 2000), replace=False)
    result = {}
    fitted = {}
    for name, algo in algos.items():
        xx = x[subset] if name in ("dbscan", "optics") else x
        labels = algo.fit_predict(xx)
        usable = labels >= 0
        unique = np.unique(labels[usable])
        score = None
        db = None
        if len(unique) > 1 and usable.sum() > len(unique):
            score = float(
                silhouette_score(
                    xx[usable],
                    labels[usable],
                    sample_size=min(1000, usable.sum()),
                    random_state=seed,
                )
            )
            db = float(davies_bouldin_score(xx[usable], labels[usable]))
        result[name] = {
            "silhouette": score,
            "davies_bouldin": db,
            "sizes": {str(k): int((labels == k).sum()) for k in np.unique(labels)},
            "fit_rows": len(xx),
            "noise_rows": int((labels == -1).sum()),
        }
        fitted[name] = algo
    km = fitted["kmeans"]
    base = km.predict(x[subset])
    ari = []
    for _ in range(3):
        idx = rng.choice(len(x), len(x), replace=True)
        m = KMeans(3, n_init=5, random_state=seed).fit(x[idx])
        ari.append(adjusted_rand_score(base, m.predict(x[subset])))
    result["kmeans"]["stability_ari"] = float(np.mean(ari))
    profiles = raw[FEATURES].copy()
    profiles["segment"] = km.labels_
    result["kmeans"]["profiles"] = (
        profiles.groupby("segment").mean().reset_index().to_dict("records")
    )
    pca = PCA(2, random_state=seed).fit(x)
    coords = pca.transform(x[subset])
    projection = pd.DataFrame(
        {
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "segment": base,
            "row_index": raw.index[subset],
        }
    )
    return result, km, pca, projection


def fairness(df, y, p, threshold):
    rows = []
    groups = {
        "SEX": df.SEX,
        "age_group": pd.cut(
            df.AGE, [17, 30, 45, 60, 110], labels=["18-30", "31-45", "46-60", "61+"]
        ),
    }
    for field, series in groups.items():
        for group in series.dropna().unique():
            mask = (series == group).to_numpy()
            yy = np.asarray(y)[mask]
            pp = np.asarray(p)[mask] >= threshold
            if len(yy) < 30:
                rows.append(
                    {
                        "field": field,
                        "group": str(group),
                        "n": len(yy),
                        "status": "insufficient sample",
                    }
                )
                continue
            tn, fp, fn, tp = confusion_matrix(yy, pp, labels=[0, 1]).ravel()
            rows.append(
                {
                    "field": field,
                    "group": str(group),
                    "n": len(yy),
                    "fpr": float(fp / (fp + tn)) if fp + tn else None,
                    "fnr": float(fn / (fn + tp)) if fn + tp else None,
                    "default_rate": float(yy.mean()),
                }
            )
    return rows


def train(
    df,
    output,
    source="UCI 350",
    seed=42,
    quick=False,
    explain=True,
    evaluate_test=False,
):
    from xgboost import XGBClassifier

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "TEST_EVALUATED.json").exists():
        raise FileExistsError(
            "Final test is already evaluated; preserve experiment directory"
        )
    tr, va, te = split_data(df, seed)
    x = tr[FEATURES]
    y = tr[TARGET]
    vx = va[FEATURES]
    vy = va[TARGET]
    prep = preprocessor().fit(x)
    z = prep.transform(x)
    cluster_metrics, km, pca, projection = clustering(z, tr, seed, quick)
    projection.to_csv(out / "projection.csv", index=False)
    candidates = {
        "dummy": DummyClassifier(strategy="prior"),
        "logreg": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        ),
        "tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=10, random_state=seed
        ),
        "knn": KNeighborsClassifier(n_neighbors=25),
        "random_forest": RandomForestClassifier(
            n_estimators=30 if quick else 120,
            max_depth=8,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=seed,
        ),
        "xgboost": XGBClassifier(
            n_estimators=30 if quick else 120,
            max_depth=3,
            learning_rate=0.07,
            eval_metric="logloss",
            n_jobs=1,
            random_state=seed,
        ),
    }
    models = {}
    validation = {}
    probabilities = {}
    for name, est in candidates.items():
        pipe = Pipeline([("prepare", preprocessor()), ("model", est)])
        # Each calibration fold independently fits preprocessing on its own training partition.
        model = (
            pipe
            if name == "dummy"
            else CalibratedClassifierCV(pipe, method="sigmoid", cv=3)
        )
        model.fit(x, y)
        p = model.predict_proba(vx)[:, 1]
        probabilities[name] = p
        models[name] = model
        validation[name] = classification_metrics(vy, p, 0.5)
    selected = max(
        (k for k in models if k != "dummy"), key=lambda k: validation[k]["roc_auc"]
    )
    model = models[selected]

    # Frozen train-fitted segmenter. One-hot segment representation avoids ordinal assumptions.
    def augment(frame):
        labels = km.predict(prep.transform(frame))
        return np.column_stack([prep.transform(frame), np.eye(3)[labels]])

    aug = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, random_state=seed), cv=3
    ).fit(augment(x), y)
    # Compare matched logistic baselines so model-family changes cannot masquerade as a cluster gain.
    base = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, random_state=seed), cv=3
    ).fit(prep.transform(x), y)
    pa = aug.predict_proba(augment(vx))[:, 1]
    pb = base.predict_proba(prep.transform(vx))[:, 1]
    rng = np.random.default_rng(seed)
    deltas = []
    target = vy.to_numpy()
    for _ in range(100):
        idx = rng.integers(0, len(vy), len(vy))
        if len(np.unique(target[idx])) == 2:
            deltas.append(
                roc_auc_score(target[idx], pa[idx])
                - roc_auc_score(target[idx], pb[idx])
            )
    ci = np.quantile(deltas, [0.025, 0.975]).tolist()
    # The segment is exploratory in this release; scoring remains the selected calibrated original-feature model.
    ablation = {
        "auc_delta": float(roc_auc_score(vy, pa) - roc_auc_score(vy, pb)),
        "auc_delta_ci95": ci,
        "segment_retained": False,
        "scope": "Validation matched logistic models; preprocessing and clustering fit on train. Nested ablation CV is not claimed.",
    }
    p = probabilities[selected]
    thresholds = np.linspace(0.05, 0.95, 181)
    threshold = float(
        max(thresholds, key=lambda t: f1_score(vy, p >= t, zero_division=0))
    )
    bundle = {
        "model": model,
        "models": models,
        "preprocessor": prep,
        "segmenter": km,
        "pca": pca,
        "features": FEATURES,
        "threshold": threshold,
        "selected": selected,
        "source": source,
        "train_indices": tr.index.tolist(),
        "background": x.iloc[: min(40, len(x))],
        "dataset_sha256": hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(),
    }
    joblib.dump(bundle, out / "model.joblib")
    result = {
        "source": source,
        "rows": len(df),
        "seed": seed,
        "dataset_sha256": bundle["dataset_sha256"],
        "split": {
            k: len(d) for k, d in zip(["train", "validation", "test"], [tr, va, te])
        },
        "selected_model": selected,
        "threshold": threshold,
        "validation": validation,
        "clustering": cluster_metrics,
        "ablation": ablation,
        "test": None,
        "limitations": [
            "Taiwan 2005 sample, not representative of present-day banking.",
            "Probabilities are academic estimates; no automatic credit decision.",
            "AUC targets are goals, not guaranteed outcomes.",
        ],
    }
    if evaluate_test:
        with (out / "TEST_EVALUATED.json").open("x") as f:
            json.dump(
                {
                    "selected": selected,
                    "threshold": threshold,
                    "dataset": bundle["dataset_sha256"],
                },
                f,
            )
        test_metrics = {
            k: classification_metrics(
                te[TARGET], m.predict_proba(te[FEATURES])[:, 1], threshold
            )
            for k, m in models.items()
        }
        result["test"] = {
            "selected": test_metrics[selected],
            "all_frozen_models": test_metrics,
        }
        result["fairness"] = fairness(
            te, te[TARGET], model.predict_proba(te[FEATURES])[:, 1], threshold
        )
    if explain:
        from .explain import explain_rows, global_importance

        result["explanations"] = explain_rows(bundle, vx.iloc[:5])
        result["permutation_importance"] = global_importance(
            bundle, vx.iloc[: min(500, len(vx))], vy.iloc[: min(500, len(vy))]
        )
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def load_artifact(output):
    # joblib is executable serialization: only load locally trained, trusted artifacts.
    return joblib.load(Path(output) / "model.joblib")


def predict(bundle, rows):
    validate_data(rows, require_target=False)
    p = bundle["model"].predict_proba(rows[bundle["features"]])[:, 1]
    return {
        "probability": p.tolist(),
        "threshold": bundle["threshold"],
        "prediction": (p >= bundle["threshold"]).astype(int).tolist(),
        "segment": bundle["segmenter"]
        .predict(bundle["preprocessor"].transform(rows[bundle["features"]]))
        .tolist(),
        "model": bundle["selected"],
        "source": bundle["source"],
    }

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from .data import FEATURES


def explain_rows(bundle, rows):
    import shap
    from lime.lime_tabular import LimeTabularExplainer

    model = bundle["model"]
    background = bundle["background"][FEATURES]

    def predict_np(a):
        return model.predict_proba(pd.DataFrame(a, columns=FEATURES))

    explainer = shap.Explainer(
        lambda a: predict_np(a)[:, 1],
        background.to_numpy(),
        algorithm="permutation",
        feature_names=FEATURES,
        seed=42,
    )
    values = explainer(rows[FEATURES].to_numpy(), max_evals=2 * len(FEATURES) + 1)
    lime = LimeTabularExplainer(
        background.to_numpy(),
        feature_names=FEATURES,
        class_names=["no default", "default"],
        mode="classification",
        random_state=42,
        discretize_continuous=True,
    )
    result = []
    for j, (_, row) in enumerate(rows.iterrows()):
        local = lime.explain_instance(
            row[FEATURES].to_numpy(), predict_np, num_features=8, num_samples=500
        )
        result.append(
            {
                "probability": float(predict_np(row[FEATURES].to_numpy()[None])[0, 1]),
                "shap_base": float(values.base_values[j]),
                "shap": dict(zip(FEATURES, values.values[j].astype(float).tolist())),
                "lime": [(str(k), float(v)) for k, v in local.as_list()],
                "interpretation": "Contributions to academic default probability; LIME is a local surrogate, disagreements possible.",
            }
        )
    return result


def global_importance(bundle, x, y):
    r = permutation_importance(
        bundle["model"],
        x[FEATURES],
        y,
        n_repeats=3,
        scoring="roc_auc",
        random_state=42,
        n_jobs=1,
    )
    return [
        {"feature": k, "importance": float(v), "std": float(s)}
        for k, v, s in zip(FEATURES, r.importances_mean, r.importances_std)
    ]

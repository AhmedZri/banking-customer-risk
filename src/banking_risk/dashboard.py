import json
import os
from pathlib import Path
import secrets
import pandas as pd
import plotly.express as px
import streamlit as st
from banking_risk.data import FEATURES, synthetic_customers
from banking_risk.training import load_artifact, predict

st.set_page_config(
    page_title="Banking Risk Intelligence", page_icon="🏦", layout="wide"
)
st.title("Banking Risk · Analyst Workbench")
st.caption("Customer segmentation, calibrated default probabilities and explainability")
key = os.getenv("API_TOKEN", "")
entered = st.sidebar.text_input("Local access token", type="password")
if not key or not secrets.compare_digest(entered, key):
    st.info("Set API_TOKEN, then enter it in the sidebar.")
    st.stop()
root = Path(st.sidebar.text_input("Artifacts", "artifacts/uci"))
if not (root / "metrics.json").exists():
    st.info("Run the training CLI first.")
    st.stop()
m = json.loads((root / "metrics.json").read_text())
st.sidebar.info("Source: " + m["source"])
page = st.sidebar.radio(
    "Module",
    [
        "1 · Data",
        "2 · Preparation",
        "3 · Segmentation",
        "4 · Credit scoring",
        "5 · Explainability",
        "6 · Individual analysis",
    ],
)
if page.startswith("1"):
    st.metric("Dataset rows", m["rows"])
    st.json(m["split"])
    path = Path(st.text_input("Local dataset", "data/uci.csv"))
    if path.exists():
        df = pd.read_csv(path)
        st.dataframe(df.describe().T, width="stretch")
        st.plotly_chart(
            px.histogram(df, x="default", title="Observed class distribution"),
            width="stretch",
        )
        st.plotly_chart(
            px.imshow(df[FEATURES].corr(), title="Feature correlations"),
            width="stretch",
        )
else:
    if page.startswith("2"):
        st.markdown(
            "**60/20/20 stratified split.** Imputation, encoders and scaling are fit inside each training/calibration fold. Final test selection is prohibited."
        )
        st.json(
            {
                "dataset_sha256": m["dataset_sha256"],
                "seed": m["seed"],
                "threshold": m["threshold"],
            }
        )
        st.json(m["limitations"])
    elif page.startswith("3"):
        projection = pd.read_csv(root / "projection.csv")
        st.plotly_chart(
            px.scatter(
                projection,
                x="pc1",
                y="pc2",
                color=projection.segment.astype(str),
                title="PCA projection · train-only fit",
            ),
            width="stretch",
        )
        algo = st.selectbox("Algorithm", list(m["clustering"]))
        st.json(m["clustering"][algo])
        st.warning(
            "Density algorithms are evaluated on a documented train subset. A 2D projection is not evidence of cluster quality."
        )
        st.json(m["ablation"])
        with st.expander("Explore training-only segmentation parameters"):
            algo = st.selectbox(
                "Exploratory algorithm", ["kmeans", "gmm", "dbscan", "optics"]
            )
            projection = st.selectbox("2D method", ["pca", "tsne", "umap"])
            k = st.slider("Clusters", 2, 8, 3)
            eps = st.slider("DBSCAN epsilon", 0.1, 8.0, 2.5)
            if st.button("Run training exploration"):
                from banking_risk.exploration import explore

                bundle = load_artifact(root)
                data_path = Path("data/uci.csv")
                if not data_path.exists():
                    st.error("Download the UCI dataset first.")
                else:
                    train = (
                        pd.read_csv(data_path)
                        .loc[bundle["train_indices"], FEATURES]
                        .sample(
                            n=min(1500, len(bundle["train_indices"])), random_state=42
                        )
                    )
                    result = explore(train, algo, projection, k, eps)
                    st.plotly_chart(
                        px.scatter(
                            pd.DataFrame(result["projection"]),
                            x="x",
                            y="y",
                            color="segment",
                        ),
                        width="stretch",
                    )
                    st.json(
                        {
                            key: value
                            for key, value in result.items()
                            if key != "projection"
                        }
                    )
    elif page.startswith("4"):
        rows = [
            {
                "model": k,
                **{
                    metric: v[metric]
                    for metric in ["roc_auc", "pr_auc", "brier", "ece", "f1"]
                },
            }
            for k, v in m["validation"].items()
        ]
        st.dataframe(rows, width="stretch")
        model = st.selectbox(
            "Model",
            list(m["validation"]),
            index=list(m["validation"]).index(m["selected_model"]),
        )
        r = m["validation"][model]
        a, b = st.columns(2)
        a.plotly_chart(
            px.line(
                x=r["roc"]["fpr"],
                y=r["roc"]["tpr"],
                title="Validation ROC",
                labels={"x": "FPR", "y": "TPR"},
            ),
            width="stretch",
        )
        b.plotly_chart(
            px.line(
                x=r["pr"]["recall"],
                y=r["pr"]["precision"],
                title="Validation precision-recall",
            ),
            width="stretch",
        )
        st.plotly_chart(
            px.line(
                x=r["calibration"]["predicted"],
                y=r["calibration"]["observed"],
                title="Calibration",
            ),
            width="stretch",
        )
        st.plotly_chart(
            px.imshow(
                r["confusion"], text_auto=True, title="Confusion matrix · threshold 0.5"
            ),
            width="stretch",
        )
        st.caption(
            "The selected operational threshold was fixed on validation. This comparison displays 0.5 for all candidates."
        )
        st.json(m["test"])
        st.dataframe(m.get("fairness", []), width="stretch")
    elif page.startswith("5"):
        items = m.get("explanations", [])
        if not items:
            st.info("Train without --skip-explain to generate explanations.")
        else:
            e = items[st.slider("Case", 0, len(items) - 1, 0)]
            st.metric("Estimated default probability", f"{e['probability']:.1%}")
            contributions = pd.DataFrame(
                {"feature": list(e["shap"]), "contribution": list(e["shap"].values())}
            ).sort_values("contribution")
            st.plotly_chart(
                px.bar(
                    contributions,
                    x="contribution",
                    y="feature",
                    orientation="h",
                    title="SHAP probability contributions",
                ),
                width="stretch",
            )
            st.dataframe(
                e["lime"], column_config={"0": "LIME condition", "1": "Local weight"}
            )
            st.caption(e["interpretation"])
            st.dataframe(m.get("permutation_importance", []))
            import plotly.graph_objects as go

            ranked = sorted(
                e["shap"].items(), key=lambda item: abs(item[1]), reverse=True
            )
            top = ranked[:8]
            rest = sum(v for _, v in ranked[8:])
            fig = go.Figure(
                go.Waterfall(
                    x=["Base"]
                    + [k for k, _ in top]
                    + ["Other features", "Probability"],
                    y=[e["shap_base"]] + [v for _, v in top] + [rest, 0],
                    measure=["absolute"] + ["relative"] * (len(top) + 1) + ["total"],
                )
            )
            fig.update_layout(title="SHAP waterfall · probability scale")
            st.plotly_chart(fig, width="stretch")
    else:
        st.warning(
            "Fictitious input only. Academic demonstration, not a credit decision."
        )
        template = synthetic_customers(10)[FEATURES].iloc[0].to_dict()
        text = st.text_area(
            "23 feature fields (JSON)", json.dumps(template, indent=2), height=300
        )
        if st.button("Estimate risk"):
            try:
                rows = pd.DataFrame([json.loads(text)])
                bundle = load_artifact(root)
                st.json(predict(bundle, rows))
            except (ValueError, KeyError, TypeError) as exc:
                st.error(str(exc))
    st.download_button(
        "Export experiment metrics", json.dumps(m, indent=2), "metrics.json"
    )

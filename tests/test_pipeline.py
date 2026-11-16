import numpy as np
import pytest
from banking_risk.data import synthetic_customers, FEATURES, TARGET, split_data
from banking_risk.training import train, load_artifact, predict, classification_metrics


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    p = tmp_path_factory.mktemp("banking")
    data = synthetic_customers(500)
    result = train(
        data, p, source="synthetic", quick=True, explain=False, evaluate_test=True
    )
    return data, p, result


def test_all_models_and_metrics(trained):
    _, _, r = trained
    assert set(r["validation"]) == {
        "dummy",
        "logreg",
        "tree",
        "knn",
        "random_forest",
        "xgboost",
    }
    assert set(r["clustering"]) >= {"kmeans", "gmm", "dbscan", "optics"}
    assert r["test"]["selected"]["roc_auc"] >= 0
    assert "auc_delta_ci95" in r["ablation"]


def test_inference_parity_and_train_only_scaler(trained):
    data, p, r = trained
    bundle = load_artifact(p)
    row = data[FEATURES].iloc[[0]]
    prediction = predict(bundle, row)
    assert 0 <= prediction["probability"][0] <= 1
    assert prediction["probability"][0] == pytest.approx(
        bundle["model"].predict_proba(row)[0, 1]
    )
    tr, va, te = split_data(data)
    assert set(bundle["train_indices"]) == set(tr.index)
    assert not set(bundle["train_indices"]) & set(te.index)
    assert TARGET not in bundle["features"]


def test_sealed_test(trained):
    data, p, _ = trained
    with pytest.raises(FileExistsError):
        train(
            data, p, source="synthetic", quick=True, explain=False, evaluate_test=True
        )


def test_known_metrics():
    r = classification_metrics(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 0.5
    )
    assert r["roc_auc"] == 1 and r["gini"] == 1 and r["ks"] == 1
    assert r["f1"] == 1 and 0 <= r["ece"] <= 1


def test_optional_mlflow_script_runs_locally(trained, tmp_path):
    import subprocess, sys
    from pathlib import Path

    _, artifacts, _ = trained
    script = Path(__file__).resolve().parents[1] / "scripts/log_experiment.py"
    result = subprocess.run(
        [sys.executable, str(script), "--artifacts", str(artifacts)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "mlruns").exists()

"""Optional local MLflow logging, no training or test reopening."""

import os
import argparse, json
from pathlib import Path
import mlflow

p = argparse.ArgumentParser()
p.add_argument("--artifacts", default="artifacts/uci")
args = p.parse_args()
root = Path(args.artifacts)
m = json.loads((root / "metrics.json").read_text())
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
mlflow.set_experiment("ensi-banking-risk")
with mlflow.start_run():
    mlflow.log_params(
        {
            "source": m["source"],
            "seed": m["seed"],
            "dataset_sha256": m["dataset_sha256"],
            "selected_model": m["selected_model"],
        }
    )
    mlflow.log_metrics(
        {"validation_auc": m["validation"][m["selected_model"]]["roc_auc"]}
    )
    mlflow.log_artifact(str(root / "metrics.json"))

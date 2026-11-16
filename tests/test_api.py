from fastapi.testclient import TestClient
from banking_risk.api import create_app
from banking_risk.data import FEATURES, synthetic_customers


def test_auth_and_missing_artifact(tmp_path):
    c = TestClient(create_app(tmp_path, token="secret"))
    assert c.get("/health").status_code == 200
    assert c.get("/metrics").status_code == 401
    assert c.get("/metrics", headers={"X-API-Key": "secret"}).status_code == 503


def test_reject_missing_extra_and_nonfinite(tmp_path):
    c = TestClient(create_app(tmp_path, token="secret"))
    h = {"X-API-Key": "secret"}
    row = synthetic_customers(10)[FEATURES].iloc[0].to_dict()
    for invalid in [
        {**row, "ID": 1},
        {k: v for k, v in row.items() if k != "AGE"},
        {**row, "AGE": -3},
    ]:
        assert (
            c.post("/predict", headers=h, json={"rows": [invalid]}).status_code == 422
        )


def test_nonfinite_rejected_without_server_error(tmp_path):
    import json

    c = TestClient(create_app(tmp_path, token="secret"))
    h = {"X-API-Key": "secret", "Content-Type": "application/json"}
    row = synthetic_customers(10)[FEATURES].iloc[0].to_dict()
    row["AGE"] = float("nan")
    assert (
        c.post("/predict", headers=h, content=json.dumps({"rows": [row]})).status_code
        == 422
    )

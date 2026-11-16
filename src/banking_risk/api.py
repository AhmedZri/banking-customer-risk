import json
import logging
import os
from pathlib import Path
import secrets
import time
import pandas as pd
from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from .data import FEATURES, validate_data
from .training import load_artifact, predict


class RowsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    rows: list[dict[str, float]] = Field(min_length=1, max_length=100)


def create_app(artifacts=None, token=None):
    root = Path(artifacts or os.getenv("ARTIFACT_DIR", "artifacts/uci"))
    key = token if token is not None else os.getenv("API_TOKEN", "")
    app = FastAPI(title="Intelligent Banking Customer Risk", version="0.1.0")
    log = logging.getLogger("banking_risk.api")
    cached = {}

    def auth(x_api_key: str | None = Header(default=None)):
        if not key:
            raise HTTPException(503, "Configure API_TOKEN")
        if not x_api_key or not secrets.compare_digest(x_api_key, key):
            raise HTTPException(401, "Invalid API key")

    def bundle():
        if not (root / "model.joblib").exists():
            raise HTTPException(503, "Train a trusted model first")
        if "bundle" not in cached:
            cached["bundle"] = load_artifact(root)
        return cached["bundle"]

    def validated(body):
        if any(set(row) != set(FEATURES) for row in body.rows):
            raise HTTPException(
                422, "Exactly 23 documented features required; ID/target are forbidden"
            )
        rows = pd.DataFrame(body.rows)
        try:
            validate_data(rows, False)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return rows

    @app.exception_handler(RequestValidationError)
    async def validation_failure(request, exc):
        # Do not echo invalid values (NaN, infinities or personal data) into JSON errors.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"loc": list(e["loc"]), "type": e["type"], "msg": e["msg"]}
                    for e in exc.errors()
                ]
            },
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "model_ready": (root / "model.joblib").exists()}

    @app.get("/metrics", dependencies=[Depends(auth)])
    def metrics():
        if not (root / "metrics.json").exists():
            raise HTTPException(503, "Train first")
        return json.loads((root / "metrics.json").read_text())

    @app.post("/predict", dependencies=[Depends(auth)])
    def predict_rows(body: RowsRequest):
        start = time.perf_counter()
        rows = validated(body)
        result = predict(bundle(), rows)
        log.info(
            json.dumps(
                {
                    "timestamp": time.time(),
                    "model": result["model"],
                    "duration_ms": (time.perf_counter() - start) * 1000,
                    "rows": len(rows),
                }
            )
        )
        return result

    @app.post("/explain", dependencies=[Depends(auth)])
    def explain(body: RowsRequest):
        rows = validated(body)
        if len(rows) > 5:
            raise HTTPException(422, "At most 5 explanation rows per request")
        from .explain import explain_rows

        return explain_rows(bundle(), rows)

    @app.get("/segments", dependencies=[Depends(auth)])
    def segments():
        return metrics()["clustering"]

    return app


app = create_app()

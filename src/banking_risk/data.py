import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = (
    [
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)
TARGET = "default"
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]


def validate_data(df, require_target=True):
    required = FEATURES + ([TARGET] if require_target else [])
    if not set(required) <= set(df):
        raise ValueError("Missing required fields: " + str(set(required) - set(df)))
    if df.empty:
        raise ValueError("Empty dataset")
    try:
        x = df[required].astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Features must be numeric coded values") from exc
    # Missing values may be imputed using train-only statistics, but infinity is invalid.
    if np.isinf(x.to_numpy()).any():
        raise ValueError("Infinite input")
    if (x[FEATURES].abs() > 1e12).any().any():
        raise ValueError("Feature magnitude exceeds documented absolute bound 1e12")
    for c in required:
        if x[c].isna().all():
            raise ValueError("Entirely missing field: " + c)
    if require_target and (x[TARGET].isna().any() or not x[TARGET].isin([0, 1]).all()):
        raise ValueError("Target must be binary")
    for c, values in {
        "SEX": [1, 2],
        "EDUCATION": list(range(7)),
        "MARRIAGE": list(range(4)),
        **{
            k: list(range(-2, 10))
            for k in FEATURES
            if k.startswith("PAY_") and not k.startswith("PAY_AMT")
        },
    }.items():
        if not x[c].dropna().isin(values).all():
            raise ValueError("Unknown coded category: " + c)
    if (
        (x.AGE.dropna() < 18).any()
        or (x.AGE.dropna() > 110).any()
        or (x.LIMIT_BAL.dropna() <= 0).any()
    ):
        raise ValueError("Out-of-domain age or limit")
    if (x[[f"PAY_AMT{i}" for i in range(1, 7)]].dropna() < 0).any().any():
        raise ValueError("Negative payment amount")
    if "ID" in df and df.ID.duplicated().any():
        raise ValueError("Duplicate customer identifiers")
    return df.copy()


def split_data(df, seed=42):
    validate_data(df)
    train, other = train_test_split(
        df, test_size=0.4, stratify=df[TARGET], random_state=seed
    )
    val, test = train_test_split(
        other, test_size=0.5, stratify=other[TARGET], random_state=seed
    )
    return train, val, test


def synthetic_customers(n=600, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=range(n))
    df["ID"] = np.arange(1, n + 1)
    df["LIMIT_BAL"] = rng.integers(1, 40, n) * 10000
    df["SEX"] = rng.integers(1, 3, n)
    df["EDUCATION"] = rng.integers(1, 5, n)
    df["MARRIAGE"] = rng.integers(1, 4, n)
    df["AGE"] = rng.integers(21, 75, n)
    for c in FEATURES:
        if c.startswith("PAY_AMT"):
            df[c] = rng.integers(0, 20000, n)
        elif c.startswith("PAY_"):
            df[c] = rng.integers(-2, 4, n)
        elif c.startswith("BILL_AMT"):
            df[c] = rng.integers(-2000, 200000, n)
    p = 1 / (1 + np.exp(-(-2 + 0.65 * df.PAY_0 + 0.25 * df.PAY_2)))
    df[TARGET] = rng.binomial(1, p)
    return df


def download(output):
    from ucimlrepo import fetch_ucirepo

    try:
        dataset = fetch_ucirepo(id=350)
        df = dataset.data.features.copy()
        df[TARGET] = dataset.data.targets.iloc[:, 0].to_numpy()
        if not set(FEATURES) <= set(df):
            raise ValueError(
                "Repository feature names differ; using official original archive"
            )
    except (ConnectionError, ValueError):
        import io, zipfile, requests

        response = requests.get(
            "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
            timeout=30,
        )
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            with archive.open("default of credit card clients.xls") as stream:
                df = pd.read_excel(stream, header=1)
        df = df.rename(columns={"default payment next month": TARGET})
    if len(df) != 30000:
        raise ValueError("Expected 30,000 UCI rows")
    validate_data(df)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    manifest = {
        "source": "UCI Default of Credit Card Clients",
        "uci_id": 350,
        "rows": len(df),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "license": "CC BY 4.0 (verify source terms)",
        "citation": "Yeh, I. (2009). Default of Credit Card Clients. UCI Machine Learning Repository. DOI:10.24432/C55S3H",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    df.to_parquet(output.with_suffix(".parquet"), index=False)
    return manifest

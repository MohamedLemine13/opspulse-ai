from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "cpu_pct",
    "memory_pct",
    "error_rate",
    "p95_latency_ms",
]

TARGET = "incident_label"

REQUIRED_COLUMNS = [
    "event_id",
    "timestamp",
    *FEATURES,
    TARGET,
]


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and return the model's input features."""

    missing = [
        column
        for column in FEATURES
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = df[FEATURES].copy()

    for column in FEATURES:

        if not pd.api.types.is_numeric_dtype(X[column]):
            raise ValueError(
                f"{column} must be numeric"
            )

        if pd.api.types.is_bool_dtype(X[column]):
            raise ValueError(
                f"{column} cannot be boolean"
            )

    if not np.isfinite(
        X.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Features contain missing or infinite values"
        )

    for column in ["cpu_pct", "memory_pct"]:

        if not X[column].between(0, 100).all():
            raise ValueError(
                f"{column} must be between 0 and 100"
            )

    if not X["error_rate"].between(0, 1).all():
        raise ValueError(
            "error_rate must be between 0 and 1"
        )

    if not (X["p95_latency_ms"] >= 0).all():
        raise ValueError(
            "Latency cannot be negative"
        )

    return X.astype("float64")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and verify the processed monitoring dataset."""

    df = pd.read_parquet(path)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Dataset is empty")

    if df["event_id"].isna().any():
        raise ValueError("Missing event IDs")

    if df["event_id"].duplicated().any():
        raise ValueError("Duplicate event IDs")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="raise",
    )

    if df["timestamp"].isna().any():
        raise ValueError("Missing timestamps")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(
            "Incident labels must be 0 or 1"
        )

    validate_features(df)

    df[TARGET] = df[TARGET].astype(int)

    return df.sort_values(
        ["timestamp", "event_id"]
    ).reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split observations without shuffling."""

    if (
        train_ratio <= 0
        or validation_ratio <= 0
        or train_ratio + validation_ratio >= 1
    ):
        raise ValueError("Invalid split ratios")

    df = df.sort_values(
        ["timestamp", "event_id"]
    ).reset_index(drop=True)

    total = len(df)

    train_end = int(total * train_ratio)

    validation_end = (
        train_end
        + int(total * validation_ratio)
    )

    train = df.iloc[:train_end].copy()

    validation = df.iloc[
        train_end:validation_end
    ].copy()

    test = df.iloc[validation_end:].copy()

    if any(
        part.empty
        for part in [train, validation, test]
    ):
        raise ValueError(
            "Dataset is too small for the requested split"
        )

    if (
        train["timestamp"].max()
        >= validation["timestamp"].min()
        or validation["timestamp"].max()
        >= test["timestamp"].min()
    ):
        raise ValueError(
            "Equal timestamps cross a split boundary"
        )

    for name, part in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:

        if part[TARGET].nunique() != 2:

            raise ValueError(
                f"{name} must contain both classes"
            )

    return train, validation, test


def get_X_y(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target."""

    X = validate_features(df)

    y = df[TARGET].astype(int).copy()

    return X, y


def summarize_dataset(df: pd.DataFrame) -> dict:
    """Return descriptive statistics for a dataset split."""

    return {
        "rows": len(df),

        "normal_events": int(
            (df[TARGET] == 0).sum()
        ),

        "incident_events": int(
            (df[TARGET] == 1).sum()
        ),

        "incident_rate": float(
            df[TARGET].mean()
        ),

        "start_timestamp": (
            df["timestamp"].min().isoformat()
        ),

        "end_timestamp": (
            df["timestamp"].max().isoformat()
        ),
    }
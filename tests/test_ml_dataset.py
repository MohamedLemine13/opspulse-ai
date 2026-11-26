import numpy as np
import pandas as pd
import pytest

from src.ml.dataset import (
    FEATURES,
    TARGET,
    load_dataset,
    validate_features,
    chronological_split,
    get_X_y,
    summarize_dataset,
)


@pytest.fixture
def dataset():

    size = 100

    return pd.DataFrame({

        "event_id": [

            f"EVT-{index:03d}"

            for index in range(size)
        ],

        "timestamp": pd.date_range(

            "2026-01-01",

            periods=size,

            freq="5min",

            tz="UTC",
        ),

        "cpu_pct": np.tile(
            [30.0, 95.0], 50
        ),

        "memory_pct": np.tile(
            [40.0, 98.0], 50
        ),

        "error_rate": np.tile(
            [0.01, 0.30], 50
        ),

        "p95_latency_ms": np.tile(
            [200.0, 2000.0], 50
        ),

        "incident_label": np.tile(
            [0, 1], 50
        ),
    })


def test_load_dataset(
    dataset,
    tmp_path,
):

    path = tmp_path / "events.parquet"

    dataset.to_parquet(
        path,

        index=False,
    )

    result = load_dataset(path)

    assert len(result) == 100

    assert result["timestamp"].is_monotonic_increasing


def test_chronological_split(dataset):

    train, validation, test = chronological_split(
        dataset
    )

    assert len(train) == 70

    assert len(validation) == 15

    assert len(test) == 15


def test_split_chronology(dataset):

    train, validation, test = chronological_split(
        dataset
    )

    assert (

        train["timestamp"].max()

        < validation["timestamp"].min()
    )

    assert (

        validation["timestamp"].max()

        < test["timestamp"].min()
    )


def test_split_no_overlap(dataset):

    train, validation, test = chronological_split(
        dataset
    )

    train_ids = set(
        train["event_id"]
    )

    validation_ids = set(
        validation["event_id"]
    )

    test_ids = set(
        test["event_id"]
    )

    assert train_ids.isdisjoint(validation_ids)

    assert train_ids.isdisjoint(test_ids)

    assert validation_ids.isdisjoint(test_ids)


def test_feature_target_separation(dataset):

    X, y = get_X_y(dataset)

    assert list(X.columns) == FEATURES

    assert TARGET not in X.columns

    assert "event_id" not in X.columns

    assert "timestamp" not in X.columns

    assert X.shape == (100, 4)

    assert len(y) == 100


def test_missing_feature(dataset):

    invalid = dataset.drop(
        columns=["cpu_pct"]
    )

    with pytest.raises(
        ValueError,

        match="Missing features",
    ):

        validate_features(invalid)


@pytest.mark.parametrize(

    "column, invalid_value",

    [
        ("cpu_pct", -1.0),

        ("cpu_pct", 101.0),

        ("memory_pct", 120.0),

        ("error_rate", -0.1),

        ("error_rate", 1.5),

        ("p95_latency_ms", -100.0),

        ("cpu_pct", float("nan")),

        ("cpu_pct", float("inf")),
    ],
)
def test_invalid_features(

    dataset,

    column,

    invalid_value,
):

    dataset.loc[
        0, column
    ] = invalid_value

    with pytest.raises(ValueError):

        validate_features(dataset)


def test_duplicate_ids(
    dataset,
    tmp_path,
):

    dataset.loc[
        1, "event_id"
    ] = dataset.loc[
        0, "event_id"
    ]

    path = tmp_path / "events.parquet"

    dataset.to_parquet(
        path,

        index=False,
    )

    with pytest.raises(

        ValueError,

        match="Duplicate event IDs",
    ):

        load_dataset(path)


def test_invalid_target(
    dataset,
    tmp_path,
):

    dataset.loc[
        0, TARGET
    ] = 2

    path = tmp_path / "events.parquet"

    dataset.to_parquet(
        path,

        index=False,
    )

    with pytest.raises(

        ValueError,

        match="Incident labels",
    ):

        load_dataset(path)


def test_dataset_summary(dataset):

    summary = summarize_dataset(dataset)

    assert summary["rows"] == 100

    assert summary["normal_events"] == 50

    assert summary["incident_events"] == 50

    assert summary["incident_rate"] == 0.5


def test_invalid_split_ratios(dataset):

    with pytest.raises(ValueError):

        chronological_split(

            dataset,

            train_ratio=0.9,

            validation_ratio=0.2,
        )


def test_timestamp_boundary_overlap(dataset):

    dataset.loc[
        70, "timestamp"
    ] = dataset.loc[
        69, "timestamp"
    ]

    with pytest.raises(

        ValueError,

        match="Equal timestamps",
    ):

        chronological_split(dataset)
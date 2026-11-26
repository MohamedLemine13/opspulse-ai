import numpy as np
import pandas as pd
import pytest

from src.ml.models import (
    create_models,
    fit_model,
    predict_labels,
    baseline_predictions,
)


@pytest.fixture
def training_data():

    X = pd.DataFrame({

        "cpu_pct": [
            25.0, 35.0, 40.0, 50.0,
            95.0, 98.0, 75.0, 85.0,
        ],

        "memory_pct": [
            40.0, 45.0, 50.0, 60.0,
            95.0, 98.0, 99.0, 85.0,
        ],

        "error_rate": [
            0.01, 0.02, 0.03, 0.01,
            0.20, 0.30, 0.40, 0.50,
        ],

        "p95_latency_ms": [
            100.0, 200.0, 250.0, 300.0,
            1500.0, 2000.0, 2500.0, 3000.0,
        ],
    })

    y = pd.Series(
        [0, 0, 0, 0, 1, 1, 1, 1]
    )

    return X, y


def test_model_creation():

    models = create_models()

    assert set(models.keys()) == {

        "logistic_regression",

        "random_forest",

        "isolation_forest",
    }


@pytest.mark.parametrize(

    "name",

    [
        "logistic_regression",

        "random_forest",

        "isolation_forest",
    ],
)
def test_model_training(
    training_data,
    name,
):

    X, y = training_data

    model = create_models()[name]

    model = fit_model(

        name,

        model,

        X,

        y,
    )

    predictions = predict_labels(

        name,

        model,

        X,
    )

    assert len(predictions) == len(X)

    assert set(
        predictions
    ).issubset({0, 1})


def test_scaler_uses_training_data(
    training_data,
):

    X_train, y_train = training_data

    model = create_models()[
        "logistic_regression"
    ]

    model.fit(
        X_train,

        y_train,
    )

    scaler = model.named_steps[
        "scaler"
    ]

    training_mean = X_train.mean().to_numpy()

    np.testing.assert_allclose(

        scaler.mean_,

        training_mean,
    )

    # Simulate different inference observations.

    X_new = X_train.copy()

    X_new["p95_latency_ms"] += 5000

    model.predict(X_new)

    # Inference must not modify the fitted scaler.

    np.testing.assert_allclose(

        scaler.mean_,

        training_mean,
    )


def test_isolation_forest_label_mapping(
    training_data,
):

    X, y = training_data

    model = create_models()[
        "isolation_forest"
    ]

    model.fit(X)

    raw_predictions = model.predict(X)

    normalized = predict_labels(

        "isolation_forest",

        model,

        X,
    )

    expected = (

        raw_predictions == -1

    ).astype(int)

    np.testing.assert_array_equal(

        normalized,

        expected,
    )


def test_baseline_predictions():

    X = pd.DataFrame({

        "cpu_pct": [
            30.0, 99.0, 80.0
        ],

        "memory_pct": [
            40.0, 60.0, 90.0
        ],

        "error_rate": [
            0.02, 0.03, 0.40
        ],

        "p95_latency_ms": [
            200.0, 400.0, 2500.0
        ],
    })

    predictions = baseline_predictions(X)

    assert predictions.tolist() == [
        0, 0, 1
    ]
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest,
)

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from src.detector import predict_incident


RANDOM_STATE = 42


def create_models() -> dict:
    """Create the three candidate ML models."""

    logistic_regression = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=4,
    )

    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination=0.10,
        random_state=RANDOM_STATE,
        n_jobs=4,
    )

    return {
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
        "isolation_forest": isolation_forest,
    }


def fit_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """Fit supervised or unsupervised models."""

    if name == "isolation_forest":

        # Unsupervised training:
        # the algorithm does not receive incident labels.

        model.fit(X_train)

    else:

        # Supervised training

        model.fit(
            X_train,
            y_train,
        )

    return model


def predict_labels(
    name: str,
    model,
    X: pd.DataFrame,
) -> np.ndarray:
    """Normalize predictions to incident labels 0 and 1."""

    predictions = model.predict(X)

    if name == "isolation_forest":

        return (
            predictions == -1
        ).astype(int)

    return np.asarray(
        predictions,
        dtype=int,
    )


def baseline_predictions(
    X: pd.DataFrame,
) -> np.ndarray:
    """Run the original Stage 1 rule-based detector."""

    predictions = []

    for row in X.to_dict(orient="records"):

        event = {
            "error_rate": row["error_rate"],

            "latency_ms": row["p95_latency_ms"],

            "cpu_usage": row["cpu_pct"],
        }

        predictions.append(
            predict_incident(event)
        )

    return np.asarray(
        predictions,
        dtype=int,
    )
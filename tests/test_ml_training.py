import json

import pandas as pd
import pytest

from src.generate_raw_data import generate_valid_events

from src.prepare_data import prepare_dataframe

from src.ml.train import (
    run_training,
    select_best_model,
)

from src.ml.predict import IncidentPredictor


@pytest.fixture(scope="module")
def trained_artifacts(
    tmp_path_factory,
):

    root = tmp_path_factory.mktemp(
        "ml_training"
    )

    input_path = root / "events.parquet"

    models_dir = root / "models"

    reports_dir = root / "reports"

    df = prepare_dataframe(

        generate_valid_events()
    )

    df.to_parquet(

        input_path,

        index=False,

        coerce_timestamps="us",
    )

    metadata = run_training(

        input_path,

        models_dir,

        reports_dir,
    )

    return {

        "models_dir": models_dir,

        "reports_dir": reports_dir,

        "metadata": metadata,
    }


def test_model_selection():

    results = pd.DataFrame([

        {
            "model": "rule_based",
            "f1_score": 0.70,
            "recall": 0.70,
            "precision": 0.70,
        },

        {
            "model": "logistic_regression",
            "f1_score": 0.80,
            "recall": 0.80,
            "precision": 0.80,
        },

        {
            "model": "random_forest",
            "f1_score": 0.90,
            "recall": 0.90,
            "precision": 0.90,
        },
    ])

    assert select_best_model(
        results
    ) == "random_forest"


def test_baseline_not_selected_as_ml_model():

    results = pd.DataFrame([

        {
            "model": "rule_based",
            "f1_score": 0.95,
            "recall": 0.95,
            "precision": 0.95,
        },

        {
            "model": "random_forest",
            "f1_score": 0.85,
            "recall": 0.85,
            "precision": 0.85,
        },
    ])

    assert select_best_model(
        results
    ) == "random_forest"


def test_training_artifacts(
    trained_artifacts,
):

    models_dir = trained_artifacts[
        "models_dir"
    ]

    reports_dir = trained_artifacts[
        "reports_dir"
    ]

    assert (
        models_dir / "incident_model.joblib"
    ).exists()

    assert (
        models_dir / "model_metadata.json"
    ).exists()

    for filename in [

        "dataset_summary.json",

        "validation_metrics.csv",

        "test_metrics.csv",

        "test_predictions.csv",
    ]:

        assert (
            reports_dir / filename
        ).exists()


def test_training_split_metadata(
    trained_artifacts,
):

    metadata = trained_artifacts[
        "metadata"
    ]

    assert metadata["training_rows"] == 700

    assert metadata["validation_rows"] == 150

    assert metadata["test_rows"] == 150

    assert metadata["split_strategy"] == (
        "chronological_70_15_15"
    )


def test_validation_results(
    trained_artifacts,
):

    reports_dir = trained_artifacts[
        "reports_dir"
    ]

    results = pd.read_csv(

        reports_dir / "validation_metrics.csv"
    )

    assert len(results) == 4

    assert results["f1_score"].between(
        0, 1
    ).all()


def test_final_test_results(
    trained_artifacts,
):

    reports_dir = trained_artifacts[
        "reports_dir"
    ]

    results = pd.read_csv(

        reports_dir / "test_metrics.csv"
    )

    assert len(results) == 2

    assert "rule_based" in results["model"].values

    for _, row in results.iterrows():

        assert (

            row["tp"]

            + row["tn"]

            + row["fp"]

            + row["fn"]

        ) == 150


def test_saved_model_inference(
    trained_artifacts,
):

    models_dir = trained_artifacts[
        "models_dir"
    ]

    predictor = IncidentPredictor(

        models_dir / "incident_model.joblib",

        models_dir / "model_metadata.json",
    )

    result = predictor.predict({

        "cpu_pct": 95.0,

        "memory_pct": 80.0,

        "error_rate": 0.40,

        "p95_latency_ms": 2500.0,
    })

    assert result["prediction"] in [0, 1]

    assert result["status"] in [

        "NORMAL",

        "INCIDENT",
    ]


def test_saved_predictions_reproducibility(
    trained_artifacts,
):

    models_dir = trained_artifacts[
        "models_dir"
    ]

    reports_dir = trained_artifacts[
        "reports_dir"
    ]

    predictor = IncidentPredictor(

        models_dir / "incident_model.joblib",

        models_dir / "model_metadata.json",
    )

    predictions = pd.read_csv(

        reports_dir / "test_predictions.csv"
    )

    for _, row in predictions.head(10).iterrows():

        event = {

            feature: row[feature]

            for feature in predictor.metadata[
                "features"
            ]
        }

        result = predictor.predict(event)

        assert result["prediction"] == (
            row["model_prediction"]
        )


def test_invalid_inference(
    trained_artifacts,
):

    models_dir = trained_artifacts[
        "models_dir"
    ]

    predictor = IncidentPredictor(

        models_dir / "incident_model.joblib",

        models_dir / "model_metadata.json",
    )

    with pytest.raises(ValueError):

        predictor.predict({

            "cpu_pct": 150.0,

            "memory_pct": 50.0,

            "error_rate": 0.10,

            "p95_latency_ms": 300.0,
        })


def test_model_fingerprint_validation(
    trained_artifacts,
    tmp_path,
):

    models_dir = trained_artifacts[
        "models_dir"
    ]

    metadata = trained_artifacts[
        "metadata"
    ].copy()

    metadata["model_sha256"] = "invalid_hash"

    metadata_path = tmp_path / "invalid_metadata.json"

    metadata_path.write_text(

        json.dumps(metadata),

        encoding="utf-8",
    )

    with pytest.raises(

        ValueError,

        match="fingerprint",
    ):

        IncidentPredictor(

            models_dir / "incident_model.joblib",

            metadata_path,
        )
import numpy as np
import pandas as pd
import pytest

from src.spark_config import create_spark_session

from src.generate_large_data import (
    generate_large_dataset,
)

from src.benchmark import (
    pandas_processing,
    spark_processing,
    verify_results,
    benchmark_dataset,
)


@pytest.fixture(scope="module")
def spark():

    session = create_spark_session(

        "OpsPulse Benchmark Tests",

        master="local[2]",
    )

    session.conf.set(

        "spark.sql.shuffle.partitions",

        "2",
    )

    yield session

    session.stop()


@pytest.fixture
def dataset(tmp_path):

    df = pd.DataFrame({

        "event_id": [

            "EVT-001",

            "EVT-002",

            "EVT-003",
        ],

        "timestamp": pd.to_datetime(

            [

                "2026-01-01T10:00:00Z",

                "2026-01-01T10:05:00Z",

                "2026-01-01T10:10:00Z",
            ],

            utc=True,
        ),

        "service": [

            "payment-api",

            "payment-api",

            "auth-service",
        ],

        "cpu_pct": [

            50.0, 95.0, 20.0,
        ],

        "memory_pct": [

            60.0, 85.0, 30.0,
        ],

        "failed_requests_5m": [

            20, 10, 0,
        ],

        "total_requests_5m": [

            100, 20, 0,
        ],

        "p95_latency_ms": [

            500.0, 2000.0, 0.0,
        ],

        "incident_label": [

            0, 1, 0,
        ],
    })

    path = tmp_path / "events.parquet"

    df.to_parquet(

        path,

        index=False,

        coerce_timestamps="us",
    )

    return path


def test_pandas_aggregation(dataset):

    result = pandas_processing(dataset)

    payment = result[

        result["service"] == "payment-api"

    ].iloc[0]

    assert payment["total_events"] == 2

    assert payment["incident_count"] == 1

    assert payment["avg_error_rate"] == pytest.approx(0.35)

    assert payment["global_error_rate"] == pytest.approx(0.25)


def test_spark_pandas_equivalence(

    spark,

    dataset,
):

    pandas_result = pandas_processing(
        dataset
    )

    spark_result = spark_processing(

        spark,

        dataset,
    )

    verify_results(

        pandas_result,

        spark_result,
    )


def test_detect_incorrect_results(dataset):

    pandas_result = pandas_processing(
        dataset
    )

    incorrect = pandas_result.copy()

    incorrect.loc[0, "total_events"] += 1

    with pytest.raises(AssertionError):

        verify_results(

            pandas_result,

            incorrect,
        )


def test_benchmark_execution(

    spark,

    dataset,
):

    report = benchmark_dataset(

        spark,

        dataset,

        repetitions=1,
    )

    assert report["rows"] == 3

    assert report["output_matches"] is True

    assert report["pandas_median_seconds"] > 0

    assert report["spark_median_seconds"] > 0


def test_large_dataset_generation(

    dataset,

    tmp_path,
):

    output_path = tmp_path / "large.parquet"

    generate_large_dataset(

        dataset,

        output_path,

        size=103,

        batch_size=25,
    )

    df = pd.read_parquet(output_path)

    assert len(df) == 103

    assert df["event_id"].is_unique

    assert df["cpu_pct"].between(0, 100).all()

    assert df["memory_pct"].between(0, 100).all()

    assert np.isfinite(df["cpu_pct"]).all()

    assert (

        df["failed_requests_5m"]

        <= df["total_requests_5m"]

    ).all()


def test_generation_reproducibility(

    dataset,

    tmp_path,
):

    first = tmp_path / "first.parquet"

    second = tmp_path / "second.parquet"

    generate_large_dataset(

        dataset,

        first,

        size=100,

        batch_size=25,

        seed=42,
    )

    generate_large_dataset(

        dataset,

        second,

        size=100,

        batch_size=25,

        seed=42,
    )

    pd.testing.assert_frame_equal(

        pd.read_parquet(first),

        pd.read_parquet(second),
    )
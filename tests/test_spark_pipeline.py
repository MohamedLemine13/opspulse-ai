from datetime import datetime, timezone

import pytest

from pyspark.sql import functions as F

from src.spark_config import create_spark_session

from src.spark_pipeline import (
    EVENT_SCHEMA,
    validate_schema,
    split_records,
    split_duplicates,
    build_spark_features,
    aggregate_by_service,
    aggregate_by_hour,
    run_spark_pipeline,
)


@pytest.fixture(scope="module")
def spark():

    session = create_spark_session(

        app_name="OpsPulse Spark Tests",

        master="local[2]",
    )

    session.conf.set(

        "spark.sql.shuffle.partitions",

        "2",
    )

    yield session

    session.stop()


@pytest.fixture
def events(spark):

    timestamp = datetime(

        2026, 1, 3, 14, 30,

        tzinfo=timezone.utc,
    )

    records = [

        (
            "EVT-001",
            timestamp,
            "payment-api",
            50.0,
            60.0,
            20,
            100,
            500.0,
            0,
        ),

        (
            "EVT-002",
            timestamp,
            "payment-api",
            95.0,
            85.0,
            10,
            20,
            2000.0,
            1,
        ),

        (
            "EVT-003",
            timestamp,
            "auth-service",
            20.0,
            30.0,
            0,
            0,
            0.0,
            0,
        ),
    ]

    return spark.createDataFrame(

        records,

        schema=EVENT_SCHEMA,
    )


def test_valid_schema(events):

    result = validate_schema(events)

    assert result.columns == EVENT_SCHEMA.fieldNames()


def test_missing_column(events):

    invalid = events.drop("cpu_pct")

    with pytest.raises(

        ValueError,

        match="Missing required column",
    ):

        validate_schema(invalid)


def test_incompatible_type(events):

    invalid = events.withColumn(

        "cpu_pct",

        F.col("cpu_pct").cast("string"),
    )

    with pytest.raises(

        TypeError,

        match="Invalid type",
    ):

        validate_schema(invalid)


def test_valid_records(events):

    valid, rejected = split_records(events)

    assert valid.count() == 3

    assert rejected.count() == 0


@pytest.mark.parametrize(

    "column, invalid_value",

    [

        ("cpu_pct", 150.0),

        ("memory_pct", -10.0),

        ("p95_latency_ms", -100.0),

        ("p95_latency_ms", float("inf")),

        ("cpu_pct", float("nan")),

        ("failed_requests_5m", -1),

        ("total_requests_5m", -1),

        ("incident_label", 2),

        ("service", ""),

        ("event_id", None),
    ],
)
def test_invalid_records(

    events,

    column,

    invalid_value,
):

    invalid = events.withColumn(

        column,

        F.lit(invalid_value),
    )

    valid, rejected = split_records(invalid)

    assert valid.count() == 0

    assert rejected.count() == 3


def test_request_consistency(events):

    invalid = events.withColumn(

        "failed_requests_5m",

        F.col("total_requests_5m") + 1,
    )

    valid, rejected = split_records(invalid)

    assert valid.count() == 0

    assert rejected.count() == 3


def test_duplicate_handling(events):

    duplicate = events.filter(

        F.col("event_id") == "EVT-001"
    )

    combined = events.unionByName(
        duplicate
    )

    unique, rejected = split_duplicates(
        combined
    )

    assert unique.count() == 2

    assert rejected.count() == 2

    assert (

        unique.filter(

            F.col("event_id") == "EVT-001"

        ).count() == 0
    )


def test_service_normalization(events):

    modified = events.withColumn(

        "service",

        F.lit(" Payment-API "),
    )

    valid, rejected = split_records(modified)

    assert rejected.count() == 0

    services = [

        row["service"]

        for row in valid.select(

            "service"

        ).distinct().collect()
    ]

    assert services == ["payment-api"]


def test_error_rate(events):

    result = build_spark_features(events)

    row = result.filter(

        F.col("event_id") == "EVT-001"

    ).first()

    assert row["error_rate"] == pytest.approx(0.2)


def test_zero_traffic(events):

    result = build_spark_features(events)

    row = result.filter(

        F.col("event_id") == "EVT-003"

    ).first()

    assert row["error_rate"] == 0.0

    assert row["no_traffic"] == 1


def test_temporal_features(events):

    result = build_spark_features(events)

    row = result.first()

    assert row["hour_utc"] == 14

    # January 3, 2026 was a Saturday.

    assert row["day_of_week"] == 7

    assert row["is_weekend"] == 1

    assert str(row["event_date"]) == "2026-01-03"

    assert row["event_hour"].minute == 0


def test_service_aggregation(events):

    result = aggregate_by_service(

        build_spark_features(events)
    )

    payment = result.filter(

        F.col("service") == "payment-api"

    ).first()

    assert payment["total_events"] == 2

    assert payment["incident_count"] == 1

    assert payment["avg_cpu"] == pytest.approx(72.5)

    assert payment["incident_rate"] == pytest.approx(0.5)

    assert payment["avg_error_rate"] == pytest.approx(0.35)

    assert payment["global_error_rate"] == pytest.approx(0.25)


def test_hourly_aggregation(events):

    result = aggregate_by_hour(

        build_spark_features(events)
    )

    assert result.count() == 2

    total = result.agg(

        F.sum("total_events")

    ).first()[0]

    assert total == 3


def test_complete_pipeline(

    spark,

    events,

    tmp_path,
):

    input_path = tmp_path / "input.parquet"

    output_dir = tmp_path / "spark"

    report_path = tmp_path / "report.json"

    events.write.parquet(str(input_path))

    report = run_spark_pipeline(

        spark,

        input_path,

        output_dir,

        report_path,
    )

    assert report["total_records"] == 3

    assert report["accepted_records"] == 3

    assert report["rejected_records"] == 0

    assert report_path.exists()

    processed = spark.read.parquet(

        str(output_dir / "events")
    )

    assert processed.count() == 3

    assert len(processed.columns) == 16

    partition_dir = (

        output_dir

        / "events"

        / "event_date=2026-01-03"
    )

    assert partition_dir.exists()

    services = spark.read.parquet(

        str(output_dir / "service_statistics")
    )

    assert services.count() == 2
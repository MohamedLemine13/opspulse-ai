import argparse
import json

from pathlib import Path
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    TimestampType,
    DoubleType,
    LongType,
    IntegerType,
    IntegralType,
    NumericType,
)

from src.spark_config import create_spark_session


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# -------------------------------------------------
# 1. Data contract
# -------------------------------------------------

EVENT_SCHEMA = StructType([
    StructField("event_id", StringType()),
    StructField("timestamp", TimestampType()),
    StructField("service", StringType()),
    StructField("cpu_pct", DoubleType()),
    StructField("memory_pct", DoubleType()),
    StructField("failed_requests_5m", LongType()),
    StructField("total_requests_5m", LongType()),
    StructField("p95_latency_ms", DoubleType()),
    StructField("incident_label", IntegerType()),
])


def validate_schema(df: DataFrame) -> DataFrame:
    """Validate and normalize the incoming schema."""

    actual_fields = {
        field.name: field.dataType
        for field in df.schema.fields
    }

    for field in EVENT_SCHEMA.fields:

        if field.name not in actual_fields:

            raise ValueError(
                f"Missing required column: {field.name}"
            )

        actual_type = actual_fields[field.name]

        expected_type = field.dataType

        if isinstance(expected_type, IntegralType):

            compatible = isinstance(
                actual_type,
                IntegralType,
            )

        elif isinstance(expected_type, NumericType):

            compatible = isinstance(
                actual_type,
                NumericType,
            )

        else:

            compatible = (
                actual_type == expected_type
            )

        if not compatible:

            raise TypeError(
                f"Invalid type for {field.name}: "
                f"{actual_type} instead of "
                f"{expected_type}"
            )

    return df.select(*[
        F.col(field.name).cast(
            field.dataType
        ).alias(field.name)

        for field in EVENT_SCHEMA.fields
    ])


# -------------------------------------------------
# 2. Record validation
# -------------------------------------------------

def valid_record_condition():

    non_null = reduce(
        lambda left, right: left & right,

        [
            F.col(field.name).isNotNull()
            for field in EVENT_SCHEMA.fields
        ],
    )

    cpu = F.col("cpu_pct")

    memory = F.col("memory_pct")

    latency = F.col("p95_latency_ms")

    failed = F.col("failed_requests_5m")

    total = F.col("total_requests_5m")

    return F.coalesce(

        non_null

        & (F.length(F.trim("event_id")) > 0)

        & (F.length(F.trim("service")) > 0)

        & (~F.isnan(cpu))

        & (~F.isnan(memory))

        & (~F.isnan(latency))

        & cpu.between(0, 100)

        & memory.between(0, 100)

        & (latency >= 0)

        & (latency < float("inf"))

        & (failed >= 0)

        & (total >= 0)

        & (failed <= total)

        & F.col("incident_label").isin(0, 1),

        F.lit(False),
    )


def split_records(
    df: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    """Separate valid and invalid observations."""

    condition = valid_record_condition()

    valid = df.filter(condition)

    rejected = df.filter(~condition)

    valid = (

        valid

        .withColumn(
            "event_id",
            F.trim("event_id"),
        )

        .withColumn(
            "service",
            F.lower(F.trim("service")),
        )
    )

    return valid, rejected


# -------------------------------------------------
# 3. Duplicate handling
# -------------------------------------------------

def split_duplicates(
    df: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    """
    Reject every record whose event ID occurs
    more than once.
    """

    duplicate_ids = (

        df.groupBy("event_id")

        .count()

        .filter(F.col("count") > 1)

        .select("event_id")
    )

    duplicates = df.join(
        duplicate_ids,
        on="event_id",
        how="left_semi",
    )

    unique = df.join(
        duplicate_ids,
        on="event_id",
        how="left_anti",
    )

    return unique, duplicates


# -------------------------------------------------
# 4. Feature engineering
# -------------------------------------------------

def build_spark_features(
    df: DataFrame,
) -> DataFrame:
    """Generate monitoring and temporal features."""

    return (

        df

        .withColumn(

            "error_rate",

            F.when(

                F.col("total_requests_5m") > 0,

                F.col("failed_requests_5m")
                / F.col("total_requests_5m"),

            ).otherwise(F.lit(0.0))
        )

        .withColumn(

            "no_traffic",

            F.when(
                F.col("total_requests_5m") == 0,
                1,
            ).otherwise(0)
        )

        .withColumn(
            "hour_utc",
            F.hour("timestamp"),
        )

        .withColumn(
            "event_date",
            F.to_date("timestamp"),
        )

        .withColumn(
            "day_of_week",
            F.dayofweek("timestamp"),
        )

        .withColumn(

            "is_weekend",

            F.when(

                F.col("day_of_week").isin(1, 7),

                1,

            ).otherwise(0)
        )

        .withColumn(
            "event_hour",
            F.date_trunc("hour", "timestamp"),
        )
    )


# -------------------------------------------------
# 5. Aggregations
# -------------------------------------------------

def aggregate_by_service(
    df: DataFrame,
) -> DataFrame:
    """Calculate service-level monitoring statistics."""

    return (

        df.groupBy("service")

        .agg(

            F.count("*").alias(
                "total_events"
            ),

            F.sum("incident_label").alias(
                "incident_count"
            ),

            F.avg("cpu_pct").alias(
                "avg_cpu"
            ),

            F.avg("memory_pct").alias(
                "avg_memory"
            ),

            F.avg("p95_latency_ms").alias(
                "avg_latency"
            ),

            F.avg("error_rate").alias(
                "avg_error_rate"
            ),

            F.sum("failed_requests_5m").alias(
                "total_failed_requests"
            ),

            F.sum("total_requests_5m").alias(
                "total_requests"
            ),
        )

        .withColumn(

            "incident_rate",

            F.col("incident_count")
            / F.col("total_events")
        )

        .withColumn(

            "global_error_rate",

            F.when(

                F.col("total_requests") > 0,

                F.col("total_failed_requests")
                / F.col("total_requests"),

            ).otherwise(F.lit(0.0))
        )
    )


def aggregate_by_hour(
    df: DataFrame,
) -> DataFrame:
    """Calculate hourly statistics for each service."""

    return (

        df.groupBy("service", "event_hour")

        .agg(

            F.count("*").alias(
                "total_events"
            ),

            F.sum("incident_label").alias(
                "incident_count"
            ),

            F.avg("cpu_pct").alias(
                "avg_cpu"
            ),

            F.avg("memory_pct").alias(
                "avg_memory"
            ),

            F.avg("p95_latency_ms").alias(
                "avg_latency"
            ),

            F.avg("error_rate").alias(
                "avg_error_rate"
            ),
        )
    )


# -------------------------------------------------
# 6. Pipeline orchestration
# -------------------------------------------------

def run_spark_pipeline(
    spark,
    input_path: Path,
    output_dir: Path,
    report_path: Path,
) -> dict:

    source = spark.read.parquet(
        str(input_path)
    )

    source = validate_schema(source).cache()

    valid = None
    rejected = None
    duplicates = None
    processed = None

    try:

        total_count = source.count()

        valid, rejected = split_records(source)

        valid = valid.cache()

        rejected = rejected.cache()

        invalid_count = rejected.count()

        unique, duplicates = split_duplicates(
            valid
        )

        duplicates = duplicates.cache()

        duplicate_count = duplicates.count()

        processed = build_spark_features(
            unique
        ).cache()

        accepted_count = processed.count()

        service_stats = aggregate_by_service(
            processed
        )

        hourly_stats = aggregate_by_hour(
            processed
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Store events partitioned by date.

        (

            processed

            .repartition("event_date")

            .write

            .mode("overwrite")

            .partitionBy("event_date")

            .parquet(
                str(output_dir / "events")
            )
        )

        # Store aggregated statistics.

        (

            service_stats

            .coalesce(1)

            .write

            .mode("overwrite")

            .parquet(
                str(
                    output_dir
                    / "service_statistics"
                )
            )
        )

        (

            hourly_stats

            .write

            .mode("overwrite")

            .parquet(
                str(
                    output_dir
                    / "hourly_statistics"
                )
            )
        )

        # Preserve rejected observations.

        (

            rejected

            .withColumn(
                "rejection_reason",
                F.lit("Invalid record"),
            )

            .unionByName(

                duplicates.withColumn(
                    "rejection_reason",
                    F.lit("Duplicate event ID"),
                )
            )

            .write

            .mode("overwrite")

            .parquet(
                str(output_dir / "rejected")
            )
        )

        report = {

            "total_records": total_count,

            "accepted_records": accepted_count,

            "invalid_records": invalid_count,

            "duplicate_records": duplicate_count,

            "rejected_records": (
                invalid_count + duplicate_count
            ),

            "acceptance_rate": (

                accepted_count / total_count

                if total_count > 0

                else 0.0
            ),
        }

        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_path.write_text(

            json.dumps(report, indent=4),

            encoding="utf-8",
        )

        print("\nSERVICE STATISTICS")

        service_stats.orderBy(
            "service"
        ).show(truncate=False)

        print("\nHOURLY STATISTICS")

        hourly_stats.orderBy(
            "event_hour", "service"
        ).show(10, truncate=False)

        print("\nSPARK EXECUTION PLAN")

        service_stats.explain("formatted")

        return report

    finally:

        for dataframe in [
            processed,
            duplicates,
            rejected,
            valid,
            source,
        ]:

            if dataframe is not None:

                dataframe.unpersist()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--input",

        type=Path,

        default=(
            PROJECT_ROOT
            / "data/processed/events.parquet"
        ),
    )

    args = parser.parse_args()

    spark = create_spark_session(
        "OpsPulse Spark Pipeline"
    )

    try:

        report = run_spark_pipeline(

            spark,

            input_path=args.input,

            output_dir=(
                PROJECT_ROOT / "data/spark"
            ),

            report_path=(
                PROJECT_ROOT
                / "reports/spark_quality.json"
            ),
        )

        print("\nSPARK QUALITY REPORT")

        print(json.dumps(report, indent=4))

    finally:

        spark.stop()


if __name__ == "__main__":
    main()
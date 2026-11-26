import argparse
import platform

from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd

import pyarrow.parquet as pq

from pyspark.sql import functions as F

from src.spark_config import create_spark_session

from src.spark_pipeline import (
    build_spark_features,
    aggregate_by_service,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATASET_SIZES = [
    10_000,
    100_000,
    1_000_000,
]


BENCHMARK_COLUMNS = [

    "service",

    "cpu_pct",

    "memory_pct",

    "p95_latency_ms",

    "failed_requests_5m",

    "total_requests_5m",

    "incident_label",
]


def pandas_processing(
    input_path: Path,
) -> pd.DataFrame:

    df = pd.read_parquet(

        input_path,

        columns=BENCHMARK_COLUMNS,
    )

    failed = df[
        "failed_requests_5m"
    ].to_numpy(dtype=float)

    total = df[
        "total_requests_5m"
    ].to_numpy(dtype=float)

    df["error_rate"] = np.divide(

        failed,

        total,

        out=np.zeros(len(df)),

        where=total > 0,
    )

    result = df.groupby(

        "service",

        as_index=False,

    ).agg(

        total_events=(
            "service", "size"
        ),

        incident_count=(
            "incident_label", "sum"
        ),

        avg_cpu=(
            "cpu_pct", "mean"
        ),

        avg_memory=(
            "memory_pct", "mean"
        ),

        avg_latency=(
            "p95_latency_ms", "mean"
        ),

        avg_error_rate=(
            "error_rate", "mean"
        ),

        total_failed_requests=(
            "failed_requests_5m", "sum"
        ),

        total_requests=(
            "total_requests_5m", "sum"
        ),
    )

    result["incident_rate"] = (

        result["incident_count"]

        / result["total_events"]
    )

    result["global_error_rate"] = np.divide(

        result[
            "total_failed_requests"
        ].to_numpy(dtype=float),

        result[
            "total_requests"
        ].to_numpy(dtype=float),

        out=np.zeros(len(result)),

        where=(
            result["total_requests"]
            .to_numpy() > 0
        ),
    )

    return result.sort_values(

        "service"

    ).reset_index(drop=True)


def spark_processing(
    spark,
    input_path: Path,
) -> pd.DataFrame:

    df = spark.read.parquet(
        str(input_path)
    ).select(*BENCHMARK_COLUMNS)

    df = df.withColumn(

        "error_rate",

        F.when(

            F.col("total_requests_5m") > 0,

            F.col("failed_requests_5m")

            / F.col("total_requests_5m"),

        ).otherwise(F.lit(0.0))
    )

    result = aggregate_by_service(
        df
    ).orderBy("service")

    # collect() materializes the Spark query.
    # Only the small service summary is collected.

    rows = result.collect()

    return pd.DataFrame(

        [row.asDict() for row in rows],

        columns=result.columns,
    )


def verify_results(
    pandas_result: pd.DataFrame,
    spark_result: pd.DataFrame,
):

    pandas_result = pandas_result.sort_values(

        "service"

    ).reset_index(drop=True)

    spark_result = spark_result.sort_values(

        "service"

    ).reset_index(drop=True)

    # String representations must match.

    assert (

        pandas_result["service"].tolist()

        == spark_result["service"].tolist()
    )

    integer_columns = [

        "total_events",

        "incident_count",

        "total_failed_requests",

        "total_requests",
    ]

    for column in integer_columns:

        np.testing.assert_array_equal(

            pandas_result[column].to_numpy(),

            spark_result[column].to_numpy(),
        )

    float_columns = [

        "avg_cpu",

        "avg_memory",

        "avg_latency",

        "avg_error_rate",

        "incident_rate",

        "global_error_rate",
    ]

    for column in float_columns:

        np.testing.assert_allclose(

            pandas_result[column].to_numpy(),

            spark_result[column].to_numpy(),

            rtol=1e-8,

            atol=1e-10,
        )


def timed_execution(function):

    start = perf_counter()

    result = function()

    elapsed = perf_counter() - start

    return result, elapsed


def benchmark_dataset(
    spark,
    input_path: Path,
    repetitions: int = 3,
) -> dict:

    if repetitions <= 0:

        raise ValueError(
            "Repetitions must be positive"
        )

    pandas_times = []

    spark_times = []

    for index in range(repetitions):

        # Alternate execution order to reduce bias.

        if index % 2 == 0:

            pandas_result, pandas_time = (
                timed_execution(

                    lambda: pandas_processing(
                        input_path
                    )
                )
            )

            spark_result, spark_time = (
                timed_execution(

                    lambda: spark_processing(
                        spark,
                        input_path,
                    )
                )
            )

        else:

            spark_result, spark_time = (
                timed_execution(

                    lambda: spark_processing(
                        spark,
                        input_path,
                    )
                )
            )

            pandas_result, pandas_time = (
                timed_execution(

                    lambda: pandas_processing(
                        input_path
                    )
                )
            )

        verify_results(
            pandas_result,
            spark_result,
        )

        pandas_times.append(pandas_time)

        spark_times.append(spark_time)

        print(

            f"Run {index + 1}: "

            f"Pandas={pandas_time:.4f}s | "

            f"Spark={spark_time:.4f}s | "

            "Results match"
        )

    pandas_median = median(pandas_times)

    spark_median = median(spark_times)

    rows = pq.ParquetFile(
        input_path
    ).metadata.num_rows

    return {

        "dataset": input_path.name,

        "rows": rows,

        "input_size_mib": round(

            input_path.stat().st_size
            / (1024 ** 2),

            4,
        ),

        "repetitions": repetitions,

        "pandas_median_seconds": pandas_median,

        "spark_median_seconds": spark_median,

        "spark_speedup": (
            pandas_median / spark_median
        ),

        "output_matches": True,

        "pandas_runs_seconds": ";".join(
            f"{value:.6f}"
            for value in pandas_times
        ),

        "spark_runs_seconds": ";".join(
            f"{value:.6f}"
            for value in spark_times
        ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--sizes",

        type=int,

        nargs="+",

        default=DATASET_SIZES,
    )

    parser.add_argument(

        "--repetitions",

        type=int,

        default=3,
    )

    args = parser.parse_args()

    paths = [

        PROJECT_ROOT

        / "data/large"

        / f"events_{size}.parquet"

        for size in args.sizes
    ]

    for path in paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Generate the dataset first: {path}"
            )

    # Measure Spark initialization separately.

    startup_begin = perf_counter()

    spark = create_spark_session(
        "OpsPulse Benchmark"
    )

    try:

        # Trigger a first action to initialize Spark.

        spark.range(1000).count()

        startup_seconds = (

            perf_counter() - startup_begin
        )

        print(
            f"Spark startup: {startup_seconds:.4f}s"
        )

        print(
            f"Spark version: {spark.version}"
        )

        print(
            f"Execution mode: "
            f"{spark.sparkContext.master}"
        )

        print(
            f"Python version: "
            f"{platform.python_version()}"
        )

        results = []

        for path in paths:

            print(
                f"\nBENCHMARK: {path.name}"
            )

            # Untimed warm-up for each engine.

            pandas_processing(path)

            spark_processing(
                spark, path
            )

            result = benchmark_dataset(

                spark,

                path,

                repetitions=args.repetitions,
            )

            result["spark_startup_seconds"] = (
                startup_seconds
            )

            results.append(result)

        results_df = pd.DataFrame(results)

        output_path = (

            PROJECT_ROOT

            / "reports/benchmark_results.csv"
        )

        output_path.parent.mkdir(

            parents=True,

            exist_ok=True,
        )

        results_df.to_csv(

            output_path,

            index=False,
        )

        columns = [

            "rows",

            "input_size_mib",

            "pandas_median_seconds",

            "spark_median_seconds",

            "spark_speedup",

            "output_matches",
        ]

        print("\nFINAL BENCHMARK RESULTS")

        print(

            results_df[columns]

            .round(4)

            .to_string(index=False)
        )

        print(
            f"\nReport saved: {output_path}"
        )

    finally:

        spark.stop()


if __name__ == "__main__":
    main()
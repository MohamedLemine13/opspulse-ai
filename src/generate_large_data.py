import argparse

from pathlib import Path

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATASET_SIZES = [
    10_000,
    100_000,
    1_000_000,
]


BASE_COLUMNS = [
    "event_id",
    "timestamp",
    "service",
    "cpu_pct",
    "memory_pct",
    "failed_requests_5m",
    "total_requests_5m",
    "p95_latency_ms",
    "incident_label",
]


def generate_large_dataset(
    source_path: Path,
    output_path: Path,
    size: int,
    batch_size: int = 50_000,
    seed: int = 42,
):
    """Generate a larger synthetic Parquet dataset."""

    if size <= 0 or batch_size <= 0:

        raise ValueError(
            "Size and batch size must be positive"
        )

    rng = np.random.default_rng(seed)

    source = pd.read_parquet(
        source_path,
        columns=BASE_COLUMNS,
    )

    if source.empty:

        raise ValueError(
            "Source dataset is empty"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = None

    try:

        for start in range(
            0, size, batch_size
        ):

            end = min(
                start + batch_size,
                size,
            )

            current_size = end - start

            # Sample records from Stage 2.

            indices = rng.integers(

                0,

                len(source),

                size=current_size,
            )

            batch = source.iloc[
                indices
            ].copy().reset_index(drop=True)

            # Generate unique event identifiers.

            batch["event_id"] = [

                f"BIG-{index:09d}"

                for index in range(
                    start + 1,
                    end + 1,
                )
            ]

            # Introduce slight measurement variations.

            batch["cpu_pct"] = np.clip(

                batch["cpu_pct"]
                + rng.normal(
                    0, 0.5, current_size
                ),

                0,

                100,

            ).round(2)

            batch["memory_pct"] = np.clip(

                batch["memory_pct"]
                + rng.normal(
                    0, 0.5, current_size
                ),

                0,

                100,

            ).round(2)

            # Vary timestamps within a limited range.

            batch["timestamp"] = (

                batch["timestamp"]

                + pd.to_timedelta(

                    rng.integers(
                        0,
                        1440,
                        size=current_size,
                    ),

                    unit="m",
                )
            )

            # Convert the batch into an Arrow table.

            table = pa.Table.from_pandas(

                batch,

                preserve_index=False,
            )

            # Initialize the Parquet writer only once.

            if writer is None:

                writer = pq.ParquetWriter(

                    str(output_path),

                    table.schema,

                    compression="snappy",

                    coerce_timestamps="us",
                )

            writer.write_table(table)

            print(
                f"Generated {end:,}/{size:,} events"
            )

    finally:

        if writer is not None:

            writer.close()

    print(
        f"Dataset saved: {output_path}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--sizes",

        type=int,

        nargs="+",

        default=DATASET_SIZES,
    )

    args = parser.parse_args()

    source_path = (

        PROJECT_ROOT

        / "data/processed/events.parquet"
    )

    for size in args.sizes:

        output_path = (

            PROJECT_ROOT

            / "data/large"

            / f"events_{size}.parquet"
        )

        print(
            f"\nGenerating {size:,} observations"
        )

        generate_large_dataset(

            source_path,

            output_path,

            size,
        )


if __name__ == "__main__":
    main()
import json

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.validation import (
    REQUIRED_FIELDS,
    validate_event,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


REJECTED_COLUMNS = [
    "line_number",
    "event_id",
    "reason",
    "raw_record",
]


def reject_record(
    line_number: int,
    event_id,
    errors: list[str],
    raw_record: str,
) -> dict:
    """Build a rejected-record entry."""

    return {
        "line_number": line_number,
        "event_id": event_id,
        "reason": "; ".join(errors),
        "raw_record": raw_record,
    }


def read_and_validate(
    input_path: Path,
) -> tuple[list[dict], list[dict]]:
    """
    Read a JSONL file and separate valid
    records from rejected records.
    """

    valid_records = []
    rejected_records = []

    seen_ids = set()

    with input_path.open(
        "r", encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file, start=1
        ):

            raw_record = line.rstrip("\n")

            # Parse JSON

            try:

                event = json.loads(raw_record)

            except json.JSONDecodeError:

                rejected_records.append(
                    reject_record(
                        line_number,
                        None,
                        ["Invalid JSON"],
                        raw_record,
                    )
                )

                continue

            # Validate schema and field values

            errors = validate_event(event)

            event_id = (
                event.get("event_id")
                if isinstance(event, dict)
                else None
            )

            if errors:

                rejected_records.append(
                    reject_record(
                        line_number,
                        event_id,
                        errors,
                        raw_record,
                    )
                )

                continue

            # Normalize the event identifier

            event_id = event_id.strip()

            # Ensure timestamp can be represented
            # by our Pandas processing pipeline.

            timestamp = pd.to_datetime(
                event["timestamp"],
                utc=True,
                errors="coerce",
            )

            if pd.isna(timestamp):

                rejected_records.append(
                    reject_record(
                        line_number,
                        event_id,
                        ["Unsupported timestamp"],
                        raw_record,
                    )
                )

                continue

            # Detect duplicate event IDs

            if event_id in seen_ids:

                rejected_records.append(
                    reject_record(
                        line_number,
                        event_id,
                        ["Duplicate event_id"],
                        raw_record,
                    )
                )

                continue

            seen_ids.add(event_id)

            # Keep only fields in our data contract

            record = {
                field: event[field]
                for field in REQUIRED_FIELDS
            }

            record["event_id"] = event_id

            record["service"] = (
                record["service"].strip().lower()
            )

            valid_records.append(record)

    return valid_records, rejected_records


def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create derived features from validated data."""

    df = df.copy()

    failed_requests = df[
        "failed_requests_5m"
    ].to_numpy(dtype=float)

    total_requests = df[
        "total_requests_5m"
    ].to_numpy(dtype=float)

    # Feature 1: error rate
    # Safely handle observations with zero traffic.

    df["error_rate"] = np.divide(
        failed_requests,
        total_requests,

        out=np.zeros(
            len(df), dtype=float
        ),

        where=total_requests > 0,
    )

    # Feature 2: no traffic

    df["no_traffic"] = (
        df["total_requests_5m"] == 0
    ).astype("int8")

    # Feature 3: hour of day in UTC

    df["hour_utc"] = (
        df["timestamp"].dt.hour
    ).astype("int8")

    return df


def prepare_dataframe(
    records: list[dict],
) -> pd.DataFrame:
    """Convert validated records into a clean DataFrame."""

    df = pd.DataFrame(
        records,
        columns=REQUIRED_FIELDS,
    )

    # Normalize data types

    df = df.astype({
        "event_id": "string",
        "service": "string",
        "cpu_pct": "float64",
        "memory_pct": "float64",
        "failed_requests_5m": "int64",
        "total_requests_5m": "int64",
        "p95_latency_ms": "float64",
        "incident_label": "int8",
    })

    # Convert timestamp strings into UTC datetimes

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        format="ISO8601",
    )

    # Sort records in chronological order

    df = df.sort_values(
        ["timestamp", "event_id"]
    ).reset_index(drop=True)

    # Create derived features

    df = build_features(df)

    return df


def create_quality_report(
    df: pd.DataFrame,
    rejected_records: list[dict],
) -> dict:
    """Generate data-quality statistics."""

    valid_count = len(df)

    rejected_count = len(rejected_records)

    total_count = valid_count + rejected_count

    rejection_reasons = Counter()

    for record in rejected_records:

        reasons = record["reason"].split("; ")

        rejection_reasons.update(reasons)

    report = {
        "total_records": total_count,

        "valid_records": valid_count,

        "rejected_records": rejected_count,

        "acceptance_rate": (
            round(valid_count / total_count, 4)
            if total_count > 0
            else 0.0
        ),

        "rejection_rate": (
            round(rejected_count / total_count, 4)
            if total_count > 0
            else 0.0
        ),

        "normal_events": int(
            (df["incident_label"] == 0).sum()
        ),

        "incident_events": int(
            (df["incident_label"] == 1).sum()
        ),

        "no_traffic_events": int(
            df["no_traffic"].sum()
        ),

        "missing_values_after_processing": int(
            df.isna().sum().sum()
        ),

        "duplicate_ids_after_processing": int(
            df["event_id"].duplicated().sum()
        ),

        "rejection_reasons": dict(
            rejection_reasons
        ),
    }

    return report


def run_pipeline(
    input_path: Path,
    output_path: Path,
    rejected_path: Path,
    report_path: Path,
) -> dict:
    """Execute the complete data preparation pipeline."""

    # Step 1: Read and validate

    valid_records, rejected_records = (
        read_and_validate(input_path)
    )

    # Step 2: Clean and transform

    df = prepare_dataframe(valid_records)

    # Step 3: Create output directories

    for path in [
        output_path,
        rejected_path,
        report_path,
    ]:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    # Step 4: Save accepted records

    df.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    # Step 5: Save rejected records

    rejected_df = pd.DataFrame(
        rejected_records,
        columns=REJECTED_COLUMNS,
    )

    rejected_df.to_csv(
        rejected_path,
        index=False,
    )

    # Step 6: Generate the quality report

    report = create_quality_report(
        df,
        rejected_records,
    )

    with report_path.open(
        "w", encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    return report


def main():

    report = run_pipeline(

        input_path=(
            PROJECT_ROOT
            / "data/raw/events.jsonl"
        ),

        output_path=(
            PROJECT_ROOT
            / "data/processed/events.parquet"
        ),

        rejected_path=(
            PROJECT_ROOT
            / "data/processed/rejected.csv"
        ),

        report_path=(
            PROJECT_ROOT
            / "reports/data_quality.json"
        ),
    )

    print("=" * 50)
    print("OPSPULSE AI - DATA PREPARATION PIPELINE")
    print("=" * 50)

    print(
        f"Total records:    "
        f"{report['total_records']}"
    )

    print(
        f"Accepted records: "
        f"{report['valid_records']}"
    )

    print(
        f"Rejected records: "
        f"{report['rejected_records']}"
    )

    print(
        f"Acceptance rate:  "
        f"{report['acceptance_rate']:.2%}"
    )

    print(
        f"Rejection rate:   "
        f"{report['rejection_rate']:.2%}"
    )

    print("-" * 50)

    print(
        f"Normal events:    "
        f"{report['normal_events']}"
    )

    print(
        f"Incident events:  "
        f"{report['incident_events']}"
    )

    print(
        f"No traffic events: "
        f"{report['no_traffic_events']}"
    )

    print(
        f"Missing values:   "
        f"{report['missing_values_after_processing']}"
    )

    print(
        f"Duplicate IDs:    "
        f"{report['duplicate_ids_after_processing']}"
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
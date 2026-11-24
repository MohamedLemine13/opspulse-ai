import json

import pandas as pd
import pytest

from src.generate_raw_data import (
    generate_valid_events,
    generate_invalid_events,
    save_jsonl,
)

from src.prepare_data import (
    read_and_validate,
    prepare_dataframe,
    run_pipeline,
)


@pytest.fixture
def valid_event():

    return {
        "event_id": "EVT-001",
        "timestamp": "2026-01-01T08:00:00Z",
        "service": "payment-api",
        "cpu_pct": 50,
        "memory_pct": 60,
        "failed_requests_5m": 25,
        "total_requests_5m": 100,
        "p95_latency_ms": 300,
        "incident_label": 0,
    }


def test_error_rate(valid_event):

    df = prepare_dataframe(
        [valid_event]
    )

    assert df.loc[
        0, "error_rate"
    ] == pytest.approx(0.25)


def test_zero_traffic(valid_event):

    valid_event["total_requests_5m"] = 0

    valid_event["failed_requests_5m"] = 0

    df = prepare_dataframe(
        [valid_event]
    )

    assert df.loc[0, "error_rate"] == 0

    assert df.loc[0, "no_traffic"] == 1


def test_nonzero_traffic(valid_event):

    df = prepare_dataframe(
        [valid_event]
    )

    assert df.loc[0, "no_traffic"] == 0


def test_hour_feature(valid_event):

    valid_event["timestamp"] = (
        "2026-01-01T14:30:00+02:00"
    )

    df = prepare_dataframe(
        [valid_event]
    )

    # 14:30 UTC+2 corresponds to 12:30 UTC

    assert df.loc[0, "hour_utc"] == 12


def test_duplicate_detection(
    valid_event,
    tmp_path,
):

    input_path = tmp_path / "events.jsonl"

    save_jsonl(
        [valid_event, valid_event.copy()],
        input_path,
    )

    valid, rejected = read_and_validate(
        input_path
    )

    assert len(valid) == 1

    assert len(rejected) == 1

    assert (
        rejected[0]["reason"]
        == "Duplicate event_id"
    )


def test_malformed_json(
    valid_event,
    tmp_path,
):

    input_path = tmp_path / "events.jsonl"

    input_path.write_text(

        json.dumps(valid_event) + "\n"

        + '{"event_id":\n',

        encoding="utf-8",
    )

    valid, rejected = read_and_validate(
        input_path
    )

    assert len(valid) == 1

    assert len(rejected) == 1

    assert (
        rejected[0]["reason"]
        == "Invalid JSON"
    )


def test_invalid_record(
    valid_event,
    tmp_path,
):

    valid_event["cpu_pct"] = 200

    input_path = tmp_path / "events.jsonl"

    save_jsonl(
        [valid_event],
        input_path,
    )

    valid, rejected = read_and_validate(
        input_path
    )

    assert len(valid) == 0

    assert len(rejected) == 1


def test_service_normalization(
    valid_event,
    tmp_path,
):

    valid_event["service"] = (
        " Payment-API "
    )

    input_path = tmp_path / "events.jsonl"

    save_jsonl(
        [valid_event],
        input_path,
    )

    valid, rejected = read_and_validate(
        input_path
    )

    assert valid[0]["service"] == "payment-api"

    assert len(rejected) == 0


def test_chronological_sorting(valid_event):

    first = valid_event.copy()

    second = valid_event.copy()

    first["event_id"] = "EVT-002"

    first["timestamp"] = (
        "2026-01-01T10:00:00Z"
    )

    second["event_id"] = "EVT-001"

    second["timestamp"] = (
        "2026-01-01T09:00:00Z"
    )

    df = prepare_dataframe(
        [first, second]
    )

    assert df["event_id"].tolist() == [
        "EVT-001",
        "EVT-002",
    ]


def test_complete_pipeline(tmp_path):

    valid_events = generate_valid_events()

    invalid_events = generate_invalid_events(
        valid_events
    )

    input_path = tmp_path / "events.jsonl"

    output_path = tmp_path / "events.parquet"

    rejected_path = tmp_path / "rejected.csv"

    report_path = tmp_path / "report.json"

    save_jsonl(
        valid_events + invalid_events,
        input_path,
    )

    report = run_pipeline(
        input_path,
        output_path,
        rejected_path,
        report_path,
    )

    assert report["total_records"] == 1020

    assert report["valid_records"] == 1000

    assert report["rejected_records"] == 20

    assert report["normal_events"] == 700

    assert report["incident_events"] == 300

    assert report["no_traffic_events"] == 50

    assert (
        report["missing_values_after_processing"]
        == 0
    )

    assert (
        report["duplicate_ids_after_processing"]
        == 0
    )

    assert output_path.exists()

    assert rejected_path.exists()

    assert report_path.exists()

    df = pd.read_parquet(output_path)

    assert df.shape == (1000, 12)

    assert not df.isna().any().any()

    assert df["event_id"].is_unique

    assert df["timestamp"].is_monotonic_increasing

    assert df["error_rate"].between(0, 1).all()

    assert df["hour_utc"].between(0, 23).all()

    rejected_df = pd.read_csv(rejected_path)

    assert len(rejected_df) == 20

    saved_report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    assert saved_report == report


def test_pipeline_reproducibility(
    tmp_path,
):

    input_path = tmp_path / "events.jsonl"

    output_path = tmp_path / "events.parquet"

    rejected_path = tmp_path / "rejected.csv"

    report_path = tmp_path / "report.json"

    events = generate_valid_events()

    save_jsonl(events, input_path)

    first_report = run_pipeline(
        input_path,
        output_path,
        rejected_path,
        report_path,
    )

    first_df = pd.read_parquet(
        output_path
    )

    second_report = run_pipeline(
        input_path,
        output_path,
        rejected_path,
        report_path,
    )

    second_df = pd.read_parquet(
        output_path
    )

    pd.testing.assert_frame_equal(
        first_df,
        second_df,
    )

    assert first_report == second_report


def test_empty_pipeline(tmp_path):

    input_path = tmp_path / "events.jsonl"

    input_path.write_text(
        "",
        encoding="utf-8",
    )

    output_path = tmp_path / "events.parquet"

    report = run_pipeline(

        input_path,

        output_path,

        tmp_path / "rejected.csv",

        tmp_path / "report.json",
    )

    assert report["total_records"] == 0

    assert report["valid_records"] == 0

    assert report["rejected_records"] == 0

    assert report["acceptance_rate"] == 0

    df = pd.read_parquet(output_path)

    assert df.empty
import pytest

from src.validation import validate_event


@pytest.fixture
def valid_event():

    return {
        "event_id": "EVT-001",

        "timestamp": (
            "2026-01-01T08:00:00Z"
        ),

        "service": "payment-api",

        "cpu_pct": 55.0,

        "memory_pct": 65.0,

        "failed_requests_5m": 5,

        "total_requests_5m": 100,

        "p95_latency_ms": 250,

        "incident_label": 0,
    }


def test_valid_event(valid_event):

    assert validate_event(valid_event) == []


@pytest.mark.parametrize(
    "field, invalid_value",
    [
        ("cpu_pct", -1),
        ("cpu_pct", 101),
        ("cpu_pct", "unknown"),
        ("cpu_pct", True),
        ("cpu_pct", float("inf")),
        ("cpu_pct", float("nan")),

        ("memory_pct", -1),
        ("memory_pct", 101),

        ("failed_requests_5m", -1),
        ("failed_requests_5m", 1.5),

        ("total_requests_5m", -1),
        ("total_requests_5m", "100"),

        ("p95_latency_ms", -100),

        ("incident_label", 2),
        ("incident_label", -1),
        ("incident_label", True),

        ("timestamp", "invalid-date"),

        (
            "timestamp",
            "2026-01-01T08:00:00"
        ),

        ("service", ""),
        ("service", "   "),
        ("event_id", ""),

        ("cpu_pct", None),
        ("memory_pct", None),
        ("service", None),
    ],
)
def test_invalid_values(
    valid_event,
    field,
    invalid_value,
):

    valid_event[field] = invalid_value

    errors = validate_event(valid_event)

    assert len(errors) > 0


def test_missing_field(valid_event):

    del valid_event["memory_pct"]

    errors = validate_event(valid_event)

    assert "Missing field: memory_pct" in errors


def test_failed_requests_exceed_total(
    valid_event,
):

    valid_event["failed_requests_5m"] = 200

    valid_event["total_requests_5m"] = 100

    errors = validate_event(valid_event)

    assert (
        "failed_requests_5m cannot exceed "
        "total_requests_5m"
    ) in errors


def test_zero_traffic_is_valid(valid_event):

    valid_event["failed_requests_5m"] = 0

    valid_event["total_requests_5m"] = 0

    assert validate_event(valid_event) == []


def test_numeric_boundaries(valid_event):

    valid_event["cpu_pct"] = 100

    valid_event["memory_pct"] = 0

    valid_event["p95_latency_ms"] = 0

    assert validate_event(valid_event) == []


def test_non_object_record():

    errors = validate_event([1, 2, 3])

    assert errors == [
        "Record must be a JSON object"
    ]
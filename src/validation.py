import math

from datetime import datetime


REQUIRED_FIELDS = [
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


def is_number(value) -> bool:
    """Check whether a value is a finite number."""

    if type(value) is int:
        return True

    return (
        type(value) is float
        and math.isfinite(value)
    )


def validate_event(event: dict) -> list[str]:
    """
    Validate a monitoring event.

    Returns an empty list if the record is valid.
    Otherwise, returns all detected validation errors.
    """

    errors = []

    if not isinstance(event, dict):
        return ["Record must be a JSON object"]

    # 1. Check required fields

    for field in REQUIRED_FIELDS:

        if field not in event:
            errors.append(f"Missing field: {field}")

        elif event[field] is None:
            errors.append(f"Null value: {field}")

    if errors:
        return errors

    # 2. Validate string fields

    for field in ["event_id", "service"]:

        value = event[field]

        if (
            not isinstance(value, str)
            or not value.strip()
        ):

            errors.append(
                f"{field} must be a non-empty string"
            )

    # 3. Validate timestamp

    try:

        timestamp = event["timestamp"]

        if not isinstance(timestamp, str):
            raise ValueError("Invalid timestamp type")

        parsed_timestamp = datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")
        )

        if parsed_timestamp.tzinfo is None:
            raise ValueError("Missing timezone")

    except (ValueError, TypeError):

        errors.append(
            "timestamp must be a valid "
            "ISO 8601 datetime with timezone"
        )

    # 4. Validate CPU and memory

    for field in ["cpu_pct", "memory_pct"]:

        value = event[field]

        if not is_number(value):

            errors.append(
                f"{field} must be a finite number"
            )

        elif not 0 <= value <= 100:

            errors.append(
                f"{field} must be between 0 and 100"
            )

    # 5. Validate request counts

    request_fields = [
        "failed_requests_5m",
        "total_requests_5m",
    ]

    for field in request_fields:

        value = event[field]

        if type(value) is not int:

            errors.append(
                f"{field} must be an integer"
            )

        elif value < 0:

            errors.append(
                f"{field} cannot be negative"
            )

    # 6. Check request consistency

    failed = event["failed_requests_5m"]
    total = event["total_requests_5m"]

    if (
        type(failed) is int
        and type(total) is int
        and failed >= 0
        and total >= 0
    ):

        if failed > total:

            errors.append(
                "failed_requests_5m cannot exceed "
                "total_requests_5m"
            )

    # 7. Validate latency

    latency = event["p95_latency_ms"]

    if not is_number(latency):

        errors.append(
            "p95_latency_ms must be a finite number"
        )

    elif latency < 0:

        errors.append(
            "p95_latency_ms cannot be negative"
        )

    # 8. Validate incident label

    label = event["incident_label"]

    if type(label) is not int or label not in (0, 1):

        errors.append(
            "incident_label must be 0 or 1"
        )

    return errors
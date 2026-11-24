import json
import random

from datetime import datetime, timedelta, timezone
from pathlib import Path


SERVICES = [
    "payment-api",
    "auth-service",
    "user-service",
    "database-service",
    "notification-service",
]


def generate_valid_events() -> list[dict]:
    """Generate 1,000 valid monitoring events."""

    rng = random.Random(42)

    events = []

    scenarios = [
        # name, count, CPU, memory, traffic,
        # error rate, latency, label

        (
            "normal", 500,
            (10, 65), (20, 75),
            (50, 1000), (0.00, 0.03),
            (100, 500), 0
        ),

        (
            "high_traffic", 150,
            (60, 90), (50, 85),
            (1000, 5000), (0.00, 0.04),
            (400, 1300), 0
        ),

        (
            "idle", 50,
            (5, 25), (15, 45),
            (0, 0), (0, 0),
            (0, 0), 0
        ),

        (
            "service_failure", 150,
            (40, 100), (40, 95),
            (100, 2000), (0.20, 0.70),
            (1500, 4000), 1
        ),

        (
            "cpu_saturation", 100,
            (95, 100), (30, 85),
            (100, 1500), (0.01, 0.08),
            (300, 1800), 1
        ),

        (
            "memory_pressure", 50,
            (30, 85), (95, 100),
            (50, 1200), (0.02, 0.15),
            (500, 2000), 1
        ),
    ]

    for scenario in scenarios:

        (
            name,
            count,
            cpu_range,
            memory_range,
            traffic_range,
            error_range,
            latency_range,
            label,
        ) = scenario

        for _ in range(count):

            total_requests = rng.randint(
                *traffic_range
            )

            error_rate = rng.uniform(
                *error_range
            )

            failed_requests = round(
                total_requests * error_rate
            )

            event = {
                "service": rng.choice(SERVICES),

                "cpu_pct": round(
                    rng.uniform(*cpu_range), 2
                ),

                "memory_pct": round(
                    rng.uniform(*memory_range), 2
                ),

                "failed_requests_5m": failed_requests,

                "total_requests_5m": total_requests,

                "p95_latency_ms": rng.randint(
                    *latency_range
                ),

                "incident_label": label,
            }

            events.append(event)

    rng.shuffle(events)

    start_time = datetime(
        2026, 1, 1, 8, 0,
        tzinfo=timezone.utc
    )

    for index, event in enumerate(events, start=1):

        event["event_id"] = f"EVT-{index:06d}"

        event["timestamp"] = (
            start_time
            + timedelta(minutes=5 * (index - 1))
        ).isoformat()

    return events


def generate_invalid_events(
    valid_events: list[dict]
) -> list[dict]:
    """Generate deliberately invalid monitoring records."""

    invalid_events = []

    def add_invalid(changes: dict):

        event = valid_events[0].copy()

        event["event_id"] = (
            f"BAD-{len(invalid_events) + 1:03d}"
        )

        event.update(changes)

        invalid_events.append(event)

    # Invalid CPU values
    add_invalid({"cpu_pct": -10})
    add_invalid({"cpu_pct": 150})

    # Invalid memory values
    add_invalid({"memory_pct": -5})
    add_invalid({"memory_pct": 120})

    # Invalid request counts
    add_invalid({"failed_requests_5m": -1})
    add_invalid({"total_requests_5m": -100})

    # More failures than total requests
    add_invalid({
        "failed_requests_5m": 200,
        "total_requests_5m": 100,
    })

    # Invalid timestamps
    add_invalid({"timestamp": "not-a-date"})
    add_invalid({"timestamp": None})

    # Missing values
    add_invalid({"cpu_pct": None})
    add_invalid({"service": None})
    add_invalid({"service": ""})

    # Invalid labels
    add_invalid({"incident_label": 2})
    add_invalid({"incident_label": -1})

    # Invalid latency
    add_invalid({"p95_latency_ms": -500})

    # Invalid data types
    add_invalid({"cpu_pct": "unknown"})
    add_invalid({"total_requests_5m": "many"})

    # Missing required field
    event = valid_events[0].copy()

    event["event_id"] = "BAD-018"

    del event["memory_pct"]

    invalid_events.append(event)

    # Duplicate records
    invalid_events.append(
        valid_events[0].copy()
    )

    invalid_events.append(
        valid_events[1].copy()
    )

    return invalid_events


def save_jsonl(
    events: list[dict],
    output_path: Path
):
    """Write monitoring records in JSONL format."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w", encoding="utf-8"
    ) as file:

        for event in events:

            file.write(
                json.dumps(event, allow_nan=False)
                + "\n"
            )


def main():

    output_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "events.jsonl"
    )

    valid_events = generate_valid_events()

    invalid_events = generate_invalid_events(
        valid_events
    )

    all_events = valid_events + invalid_events

    save_jsonl(all_events, output_path)

    print("RAW DATA GENERATION")
    print("-" * 40)

    print(f"Valid events:   {len(valid_events)}")
    print(f"Invalid events: {len(invalid_events)}")
    print(f"Total events:   {len(all_events)}")

    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
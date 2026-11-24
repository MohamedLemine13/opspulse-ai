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


def generate_events() -> list[dict]:
    """Generate synthetic monitoring events."""

    rng = random.Random(42)

    events = []

    start_time = datetime(
        2026, 1, 1, 8, 0,
        tzinfo=timezone.utc
    )

    scenarios = [
        {
            "name": "normal",
            "count": 20,
            "error_range": (0.00, 0.05),
            "latency_range": (100, 500),
            "cpu_range": (10, 60),
            "label": 0,
        },
        {
            "name": "high_traffic",
            "count": 5,
            "error_range": (0.00, 0.05),
            "latency_range": (1100, 1400),
            "cpu_range": (60, 85),
            "label": 0,
        },
        {
            "name": "critical_failure",
            "count": 10,
            "error_range": (0.25, 0.60),
            "latency_range": (1600, 3000),
            "cpu_range": (80, 100),
            "label": 1,
        },
        {
            "name": "degraded_performance",
            "count": 10,
            "error_range": (0.12, 0.19),
            "latency_range": (600, 1400),
            "cpu_range": (50, 90),
            "label": 1,
        },
        {
            "name": "cpu_saturation",
            "count": 5,
            "error_range": (0.01, 0.05),
            "latency_range": (300, 800),
            "cpu_range": (95, 100),
            "label": 1,
        },
    ]

    for scenario in scenarios:

        for _ in range(scenario["count"]):

            event = {
                "service": rng.choice(SERVICES),

                "error_rate": round(
                    rng.uniform(*scenario["error_range"]),
                    3
                ),

                "latency_ms": rng.randint(
                    *scenario["latency_range"]
                ),

                "cpu_usage": round(
                    rng.uniform(*scenario["cpu_range"]),
                    2
                ),

                "incident_label": scenario["label"],

                "scenario": scenario["name"],
            }

            events.append(event)

    rng.shuffle(events)

    for index, event in enumerate(events, start=1):

        event["event_id"] = f"EVT-{index:03d}"

        event["timestamp"] = (
            start_time + timedelta(minutes=index - 1)
        ).isoformat()

    return events


def save_events(events: list[dict], output_path: Path):
    """Save the generated events to a JSON file."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(events, file, indent=4)


def main():

    output_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "sample_events.json"
    )

    events = generate_events()

    save_events(events, output_path)

    print(f"Generated {len(events)} monitoring events.")
    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
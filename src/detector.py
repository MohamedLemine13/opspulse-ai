import json
from pathlib import Path


def detect_incident(event: dict) -> str:
    """
    Classify a monitoring event according to predefined rules.

    Returns:
        CRITICAL, WARNING or NORMAL.
    """

    error_rate = event["error_rate"]
    latency = event["latency_ms"]

    if error_rate >= 0.20 and latency >= 1500:
        return "CRITICAL"

    elif error_rate >= 0.10 or latency >= 1000:
        return "WARNING"

    return "NORMAL"


def predict_incident(event: dict) -> int:
    """
    Convert the severity classification into
    a binary incident prediction.
    """

    severity = detect_incident(event)

    return int(severity != "NORMAL")


def load_events(file_path: str) -> list[dict]:
    """Load monitoring events from a JSON file."""

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    """Run the detector on all available events."""

    data_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "sample_events.json"
    )

    events = load_events(data_path)

    for event in events:

        severity = detect_incident(event)

        print(
            f"{event['event_id']} | "
            f"{event['service']:<22} | "
            f"{severity}"
        )


if __name__ == "__main__":
    main()

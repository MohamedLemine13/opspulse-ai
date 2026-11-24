import pandas as pd

from src.evaluate import evaluate
from src.evaluate_processed import adapt_events


def test_adapt_events():

    df = pd.DataFrame([
        {
            "event_id": "EVT-001",
            "error_rate": 0.30,
            "p95_latency_ms": 2000,
            "cpu_pct": 95,
            "incident_label": 1,
        }
    ])

    events = adapt_events(df)

    assert len(events) == 1

    assert events[0]["latency_ms"] == 2000

    assert events[0]["cpu_usage"] == 95

    assert "p95_latency_ms" not in events[0]

    results = evaluate(events)

    assert results["tp"] == 1

    assert results["precision"] == 1.0
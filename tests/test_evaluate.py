import pytest

from src.evaluate import evaluate
from src.generate_data import generate_events


def test_evaluation():

    events = [

        # True positive
        {
            "error_rate": 0.30,
            "latency_ms": 2000,
            "incident_label": 1,
        },

        # True negative
        {
            "error_rate": 0.02,
            "latency_ms": 200,
            "incident_label": 0,
        },

        # False positive
        {
            "error_rate": 0.15,
            "latency_ms": 500,
            "incident_label": 0,
        },

        # False negative
        {
            "error_rate": 0.02,
            "latency_ms": 200,
            "incident_label": 1,
        },
    ]

    results = evaluate(events)

    assert results["total"] == 4

    assert results["tp"] == 1
    assert results["tn"] == 1
    assert results["fp"] == 1
    assert results["fn"] == 1

    assert results["precision"] == pytest.approx(0.5)
    assert results["recall"] == pytest.approx(0.5)
    assert results["accuracy"] == pytest.approx(0.5)
    assert results["f1_score"] == pytest.approx(0.5)


def test_empty_dataset():

    results = evaluate([])

    assert results["total"] == 0
    assert results["precision"] == 0.0
    assert results["recall"] == 0.0
    assert results["accuracy"] == 0.0
    assert results["f1_score"] == 0.0


def test_generated_dataset():

    events = generate_events()

    assert len(events) == 50

    assert sum(
        event["incident_label"]
        for event in events
    ) == 25

    event_ids = {
        event["event_id"]
        for event in events
    }

    assert len(event_ids) == 50


def test_data_reproducibility():

    first_dataset = generate_events()
    second_dataset = generate_events()

    assert first_dataset == second_dataset


def test_baseline_performance():

    results = evaluate(generate_events())

    assert results["tp"] == 20
    assert results["tn"] == 20
    assert results["fp"] == 5
    assert results["fn"] == 5

    assert results["precision"] == pytest.approx(0.8)
    assert results["recall"] == pytest.approx(0.8)
    assert results["accuracy"] == pytest.approx(0.8)
    assert results["f1_score"] == pytest.approx(0.8)
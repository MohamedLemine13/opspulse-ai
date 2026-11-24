import pytest

from src.detector import (
    detect_incident,
    predict_incident,
)


@pytest.mark.parametrize(
    "error_rate, latency_ms, expected",
    [
        (0.02, 200, "NORMAL"),
        (0.30, 2000, "CRITICAL"),
        (0.15, 600, "WARNING"),
        (0.02, 1200, "WARNING"),

        # Boundary conditions
        (0.20, 1500, "CRITICAL"),
        (0.10, 500, "WARNING"),
        (0.02, 1000, "WARNING"),
        (0.099, 999, "NORMAL"),

        # Only one critical condition is satisfied
        (0.25, 1499, "WARNING"),
        (0.19, 2000, "WARNING"),
    ],
)
def test_detect_incident(
    error_rate,
    latency_ms,
    expected,
):

    event = {
        "error_rate": error_rate,
        "latency_ms": latency_ms,
    }

    assert detect_incident(event) == expected


@pytest.mark.parametrize(
    "error_rate, latency_ms, expected",
    [
        (0.02, 200, 0),
        (0.15, 600, 1),
        (0.30, 2000, 1),
    ],
)
def test_predict_incident(
    error_rate,
    latency_ms,
    expected,
):

    event = {
        "error_rate": error_rate,
        "latency_ms": latency_ms,
    }

    assert predict_incident(event) == expected


def test_detector_does_not_use_label():

    event = {
        "error_rate": 0.02,
        "latency_ms": 200,
        "incident_label": 1,
    }

    assert predict_incident(event) == 0


def test_cpu_is_not_used_by_baseline():

    event = {
        "error_rate": 0.02,
        "latency_ms": 200,
        "cpu_usage": 99,
    }

    assert detect_incident(event) == "NORMAL"
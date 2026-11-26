import pytest

from src.ml.metrics import calculate_metrics


def test_perfect_predictions():

    y_true = [0, 0, 1, 1]

    y_pred = [0, 0, 1, 1]

    result = calculate_metrics(

        y_true,

        y_pred,
    )

    assert result["accuracy"] == 1.0

    assert result["precision"] == 1.0

    assert result["recall"] == 1.0

    assert result["f1_score"] == 1.0


def test_mixed_predictions():

    y_true = [0, 0, 1, 1]

    y_pred = [0, 1, 0, 1]

    result = calculate_metrics(

        y_true,

        y_pred,
    )

    assert result["tp"] == 1

    assert result["tn"] == 1

    assert result["fp"] == 1

    assert result["fn"] == 1

    assert result["accuracy"] == pytest.approx(0.5)

    assert result["precision"] == pytest.approx(0.5)

    assert result["recall"] == pytest.approx(0.5)

    assert result["f1_score"] == pytest.approx(0.5)


def test_no_incident_predictions():

    y_true = [0, 0, 1, 1]

    y_pred = [0, 0, 0, 0]

    result = calculate_metrics(

        y_true,

        y_pred,
    )

    assert result["precision"] == 0.0

    assert result["recall"] == 0.0

    assert result["f1_score"] == 0.0

    assert result["accuracy"] == 0.5


def test_all_normal():

    y_true = [0, 0, 0]

    y_pred = [0, 0, 0]

    result = calculate_metrics(

        y_true,

        y_pred,
    )

    assert result["tn"] == 3

    assert result["tp"] == 0

    assert result["fp"] == 0

    assert result["fn"] == 0
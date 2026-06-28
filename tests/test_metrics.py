import pytest

from src.metrics import improvement_percentage, mean_absolute_percentage_error, regression_metrics


def test_mean_absolute_percentage_error_ignores_zero_actuals():
    result = mean_absolute_percentage_error([100, 0, 200], [90, 50, 220])
    assert result == pytest.approx(10.0)


def test_regression_metrics_returns_expected_keys():
    metrics = regression_metrics([100, 200, 300], [110, 190, 330])
    assert set(metrics) == {"mae", "rmse", "mape"}
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert metrics["mape"] > 0


def test_improvement_percentage():
    assert improvement_percentage(5.0, 3.5) == pytest.approx(30.0)

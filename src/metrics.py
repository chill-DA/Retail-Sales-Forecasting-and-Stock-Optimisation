from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mean_absolute_percentage_error(y_true, y_pred) -> float:
    """Return MAPE as a percentage, ignoring zero actual values."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    non_zero_mask = y_true != 0

    if not np.any(non_zero_mask):
        raise ValueError("MAPE is undefined when all actual values are zero.")

    return float(
        np.mean(
            np.abs(
                (y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]
            )
        )
        * 100
    )


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return MAE, RMSE and MAPE for regression forecasts."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"mae": float(mae), "rmse": rmse, "mape": mape}


def improvement_percentage(baseline_value: float, model_value: float) -> float:
    """Return percentage reduction in error from baseline to model."""
    if baseline_value == 0:
        raise ValueError("Cannot calculate improvement when baseline value is zero.")
    return float((baseline_value - model_value) / baseline_value * 100)

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .features import DATE_COL, TARGET_COL, default_feature_columns
from .metrics import improvement_percentage, regression_metrics


def time_based_split(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_COL,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Split model-ready data into train and test sets using chronological order."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    df_sorted = df.sort_values(DATE_COL).reset_index(drop=True)
    feature_columns = feature_columns or default_feature_columns(df_sorted)

    if not feature_columns:
        raise ValueError("No feature columns available for modelling.")

    n_test = int(len(df_sorted) * test_size)
    n_train = len(df_sorted) - n_test

    if n_train <= 0 or n_test <= 0:
        raise ValueError("Not enough rows for the requested train/test split.")

    train_df = df_sorted.iloc[:n_train].copy()
    test_df = df_sorted.iloc[n_train:].copy()

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    return X_train, X_test, y_train, y_test, train_df, test_df


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train the Random Forest forecasting model."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_forecasts(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate baseline and model forecasts and return metrics plus predictions."""
    if "weekly_sales_lag_1" not in X_test.columns:
        raise ValueError("Baseline evaluation requires weekly_sales_lag_1 in X_test.")

    baseline_pred = X_test["weekly_sales_lag_1"]
    model_pred = model.predict(X_test)

    baseline_metrics = regression_metrics(y_test, baseline_pred)
    model_metrics = regression_metrics(y_test, model_pred)

    comparison = pd.DataFrame(
        [
            {"model": "Baseline lag_1", **baseline_metrics},
            {"model": "Random Forest", **model_metrics},
        ]
    )

    comparison["mape_improvement_vs_baseline"] = [
        0.0,
        improvement_percentage(baseline_metrics["mape"], model_metrics["mape"]),
    ]

    predictions = test_df[[DATE_COL, "store", TARGET_COL]].copy()
    predictions["prediction_baseline"] = baseline_pred.to_numpy()
    predictions["prediction_rf"] = model_pred
    predictions["error_rf"] = predictions[TARGET_COL] - predictions["prediction_rf"]
    predictions["abs_pct_error_rf"] = (
        predictions["error_rf"].abs() / predictions[TARGET_COL]
    ) * 100

    return comparison, predictions


def save_model(model: RandomForestRegressor, path) -> None:
    """Persist the trained model to disk."""
    joblib.dump(model, path)


def load_model(path) -> RandomForestRegressor:
    """Load a persisted model from disk."""
    return joblib.load(path)

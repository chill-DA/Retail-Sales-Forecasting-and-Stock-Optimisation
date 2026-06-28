from __future__ import annotations

import pandas as pd

TARGET_COL = "weekly_sales"
GROUP_COL = "store"
DATE_COL = "date"

RENAME_MAP = {
    "store": "store",
    "date": "date",
    "weekly_sales": "weekly_sales",
    "weekly sales": "weekly_sales",
    "holiday_flag": "holiday_flag",
    "holiday flag": "holiday_flag",
    "temperature": "temperature",
    "fuel_price": "fuel_price",
    "fuel price": "fuel_price",
    "cpi": "cpi",
    "unemployment": "unemployment",
}


def _normalise_column_name(column: str) -> str:
    """Return a lower-case, underscore-separated column name."""
    return "_".join(column.strip().lower().split())


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise common Walmart dataset column names."""
    rename_dict = {}
    for column in df.columns:
        normalised = _normalise_column_name(column)
        rename_dict[column] = RENAME_MAP.get(normalised, normalised)
    return df.rename(columns=rename_dict)


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean source sales data and enforce basic types.

    Expected fields after normalisation:
    store, date, weekly_sales, holiday_flag, temperature, fuel_price, cpi, unemployment.
    """
    df = normalise_columns(df).copy()

    required = [GROUP_COL, DATE_COL, TARGET_COL]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[GROUP_COL] = pd.to_numeric(df[GROUP_COL], errors="coerce").astype("Int64")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    optional_numeric_columns = [
        "holiday_flag",
        "temperature",
        "fuel_price",
        "cpi",
        "unemployment",
    ]
    for column in optional_numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[GROUP_COL, DATE_COL, TARGET_COL]).copy()
    df = df[df[TARGET_COL] >= 0].copy()
    df[GROUP_COL] = df[GROUP_COL].astype(int)
    df = df.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features used by the forecasting model."""
    df = df.copy()
    date_series = df[DATE_COL]

    df["year"] = date_series.dt.year
    df["month"] = date_series.dt.month
    df["weekofyear"] = date_series.dt.isocalendar().week.astype(int)
    df["dayofweek"] = date_series.dt.dayofweek
    df["is_month_start"] = date_series.dt.is_month_start.astype(int)
    df["is_month_end"] = date_series.dt.is_month_end.astype(int)
    return df


def add_lag_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 4),
    rolling_window: int = 4,
) -> pd.DataFrame:
    """
    Add lag and rolling-window features per store.

    Rolling features are based on shifted sales values, so the current week's target
    is not leaked into the predictor set.
    """
    df = df.sort_values([GROUP_COL, DATE_COL]).copy()
    grouped_sales = df.groupby(GROUP_COL, group_keys=False)[TARGET_COL]

    for lag in lags:
        df[f"weekly_sales_lag_{lag}"] = grouped_sales.shift(lag)

    shifted_sales = grouped_sales.shift(1)
    df[f"rolling_mean_{rolling_window}"] = shifted_sales.groupby(df[GROUP_COL]).rolling(
        rolling_window
    ).mean().reset_index(level=0, drop=True)
    df[f"rolling_std_{rolling_window}"] = shifted_sales.groupby(df[GROUP_COL]).rolling(
        rolling_window
    ).std().reset_index(level=0, drop=True)

    return df


def build_model_frame(df: pd.DataFrame, drop_missing_history: bool = True) -> pd.DataFrame:
    """Clean raw data and produce a model-ready forecasting frame."""
    df = clean_sales_data(df)
    df = add_time_features(df)
    df = add_lag_features(df)

    feature_history_columns = [
        "weekly_sales_lag_1",
        "weekly_sales_lag_2",
        "weekly_sales_lag_4",
        "rolling_mean_4",
        "rolling_std_4",
    ]

    if drop_missing_history:
        df = df.dropna(subset=feature_history_columns).reset_index(drop=True)

    return df


def default_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return default feature columns available in the model frame."""
    candidates = [
        "store",
        "holiday_flag",
        "temperature",
        "fuel_price",
        "cpi",
        "unemployment",
        "year",
        "month",
        "weekofyear",
        "dayofweek",
        "is_month_start",
        "is_month_end",
        "weekly_sales_lag_1",
        "weekly_sales_lag_2",
        "weekly_sales_lag_4",
        "rolling_mean_4",
        "rolling_std_4",
    ]
    return [column for column in candidates if column in df.columns]

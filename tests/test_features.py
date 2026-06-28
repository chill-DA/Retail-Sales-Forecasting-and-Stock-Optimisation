import pandas as pd

from src.features import build_model_frame, clean_sales_data


def make_sample_frame():
    rows = []
    dates = pd.date_range("2022-01-07", periods=6, freq="W-FRI")
    for store in [1, 2]:
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "Store": store,
                    "Date": date.strftime("%Y-%m-%d"),
                    "Weekly_Sales": 1000 + store * 100 + idx * 10,
                    "Holiday_Flag": 0,
                    "Temperature": 50 + idx,
                    "Fuel_Price": 3.0,
                    "CPI": 220.0,
                    "Unemployment": 7.0,
                }
            )
    return pd.DataFrame(rows)


def test_clean_sales_data_normalises_columns_and_types():
    df = clean_sales_data(make_sample_frame())
    assert {"store", "date", "weekly_sales"}.issubset(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["weekly_sales"].min() >= 0


def test_build_model_frame_creates_lag_and_rolling_features_without_current_week_leakage():
    model_df = build_model_frame(make_sample_frame())

    assert "weekly_sales_lag_1" in model_df.columns
    assert "rolling_mean_4" in model_df.columns
    assert len(model_df) > 0

    first_store_row = model_df[model_df["store"] == 1].iloc[0]
    assert first_store_row["weekly_sales_lag_1"] != first_store_row["weekly_sales"]

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import (
    DEFAULT_MODEL_FILE,
    DEFAULT_REPORT_FILE,
    DEFAULT_SAMPLE_FILE,
    MODELS_DIR,
    REPORTS_DIR,
    ensure_directories,
)
from src.features import build_model_frame, default_feature_columns
from src.model import evaluate_forecasts, save_model, time_based_split, train_random_forest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the retail sales forecasting pipeline on a CSV file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_SAMPLE_FILE,
        help="Path to source CSV. Defaults to the included sample data.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of chronologically latest rows used for testing.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees for the Random Forest model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}. Download the Kaggle data or use data/sample/walmart_sample.csv."
        )

    raw_df = pd.read_csv(args.input)
    model_df = build_model_frame(raw_df)
    feature_columns = default_feature_columns(model_df)

    X_train, X_test, y_train, y_test, train_df, test_df = time_based_split(
        model_df,
        feature_columns=feature_columns,
        test_size=args.test_size,
    )

    model = train_random_forest(
        X_train,
        y_train,
        n_estimators=args.n_estimators,
    )

    comparison, predictions = evaluate_forecasts(model, X_test, y_test, test_df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    save_model(model, DEFAULT_MODEL_FILE)
    comparison.to_csv(DEFAULT_REPORT_FILE, index=False)
    predictions.to_csv(REPORTS_DIR / "forecast_predictions.csv", index=False)

    print("Pipeline complete")
    print(f"Rows after feature engineering: {len(model_df):,}")
    print(f"Training rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print(f"\nSaved model: {DEFAULT_MODEL_FILE}")
    print(f"Saved metrics: {DEFAULT_REPORT_FILE}")
    print(f"Saved predictions: {REPORTS_DIR / 'forecast_predictions.csv'}")


if __name__ == "__main__":
    main()

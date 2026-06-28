from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_SAMPLE_FILE = SAMPLE_DATA_DIR / "walmart_sample.csv"
DEFAULT_MODEL_FILE = MODELS_DIR / "random_forest_forecast.joblib"
DEFAULT_REPORT_FILE = REPORTS_DIR / "model_comparison.csv"


def ensure_directories() -> None:
    """Create project output directories used by scripts."""
    for path in (PROCESSED_DATA_DIR, REPORTS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)

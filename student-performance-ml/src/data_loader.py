from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignment_completion",
    "sleep_hours",
    "participation",
    "performance",
]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def validate_dataset(data: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data.empty:
        raise ValueError("Dataset is empty.")
    if data[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Dataset contains missing values in required columns.")

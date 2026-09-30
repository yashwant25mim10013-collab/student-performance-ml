from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURE_COLUMNS = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignment_completion",
    "sleep_hours",
    "participation",
]
TARGET_COLUMN = "performance"


def split_data(data: pd.DataFrame):
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def make_pipeline(model):
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )

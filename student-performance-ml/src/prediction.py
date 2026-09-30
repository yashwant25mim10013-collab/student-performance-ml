from pathlib import Path
import pandas as pd

from src.model_training import load_model
from src.preprocessing import FEATURE_COLUMNS


def predict_from_values(model_path: Path, values: dict):
    model = load_model(model_path)
    row = pd.DataFrame([[values[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    label = model.predict(row)[0]

    probabilities = {}
    if hasattr(model, "predict_proba"):
        probabilities = dict(
            zip(model.classes_, model.predict_proba(row)[0])
        )
    return label, probabilities

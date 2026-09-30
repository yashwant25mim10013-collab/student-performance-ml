from pathlib import Path
import pandas as pd

from src.data_loader import load_dataset, validate_dataset
from src.model_training import train_and_compare_models
from src.prediction import predict_from_values

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"


def test_dataset_loads_and_validates():
    data = load_dataset(DATA_PATH)
    validate_dataset(data)
    assert len(data) > 0
    assert "performance" in data.columns


def test_training_returns_model_and_metrics():
    data = load_dataset(DATA_PATH)
    result = train_and_compare_models(data)
    assert result["best_model"] is not None
    assert result["best_name"] in result["metrics"]
    assert 0 <= result["metrics"][result["best_name"]]["accuracy"] <= 1


def test_saved_model_prediction(tmp_path):
    data = load_dataset(DATA_PATH)
    result = train_and_compare_models(data)
    model_path = tmp_path / "model.joblib"

    import joblib
    joblib.dump(result["best_model"], model_path)

    values = {
        "study_hours": 6,
        "attendance": 85,
        "previous_score": 72,
        "assignment_completion": 90,
        "sleep_hours": 7,
        "participation": 8,
    }
    label, probabilities = predict_from_values(model_path, values)
    assert label in {"Low", "Average", "Good"}
    assert isinstance(probabilities, dict)

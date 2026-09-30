from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import split_data, make_pipeline
from src.evaluation import evaluate_model


def get_models():
    return {
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=1000, random_state=42)
        ),
        "Decision Tree": make_pipeline(
            DecisionTreeClassifier(max_depth=5, random_state=42)
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
        ),
    }


def train_and_compare_models(data):
    X_train, X_test, y_train, y_test = split_data(data)
    metrics = {}
    trained_models = {}

    for name, model in get_models().items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        metrics[name] = evaluate_model(model, X_test, y_test)

    best_name = max(
        metrics,
        key=lambda name: (
            metrics[name]["f1_weighted"],
            metrics[name]["accuracy"],
        ),
    )

    return {
        "best_name": best_name,
        "best_model": trained_models[best_name],
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
    }


def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)

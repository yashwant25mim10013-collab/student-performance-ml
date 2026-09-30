from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_weighted": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
    }


def save_confusion_matrix(model, X_test, y_test, path: Path):
    predictions = model.predict(X_test)
    labels = sorted(set(y_test))
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot()
    plt.title("Student Performance Classification")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()

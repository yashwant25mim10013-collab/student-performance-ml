from pathlib import Path

from src.data_loader import load_dataset, validate_dataset
from src.evaluation import evaluate_model, save_confusion_matrix
from src.model_training import train_and_compare_models, save_model
from src.prediction import predict_from_values
from src.utils import print_metrics


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"


def train_flow():
    data = load_dataset(DATA_PATH)
    validate_dataset(data)

    result = train_and_compare_models(data)
    best_name = result["best_name"]
    best_model = result["best_model"]
    metrics = result["metrics"]

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "student_performance_model.joblib"
    save_model(best_model, model_path)

    print("\nModel comparison:")
    for name, values in metrics.items():
        print_metrics(name, values)

    print(f"Selected model: {best_name}")
    print(f"Saved model to: {model_path}")


def evaluation_flow():
    data = load_dataset(DATA_PATH)
    validate_dataset(data)
    result = train_and_compare_models(data)
    best_name = result["best_name"]
    best_model = result["best_model"]
    X_test = result["X_test"]
    y_test = result["y_test"]

    metrics = evaluate_model(best_model, X_test, y_test)
    print_metrics(best_name, metrics)

    OUTPUT_DIR.mkdir(exist_ok=True)
    matrix_path = OUTPUT_DIR / "confusion_matrix.png"
    save_confusion_matrix(best_model, X_test, y_test, matrix_path)
    print(f"Confusion matrix saved to: {matrix_path}")


def get_float(prompt, minimum, maximum):
    while True:
        try:
            value = float(input(prompt))
            if not minimum <= value <= maximum:
                print(f"Enter a value between {minimum} and {maximum}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a number.")


def prediction_flow():
    model_path = MODEL_DIR / "student_performance_model.joblib"
    if not model_path.exists():
        print("No saved model found. Please train the model first.")
        return

    values = {
        "study_hours": get_float("Study hours per day (0-24): ", 0, 24),
        "attendance": get_float("Attendance percentage (0-100): ", 0, 100),
        "previous_score": get_float("Previous exam score (0-100): ", 0, 100),
        "assignment_completion": get_float("Assignment completion (0-100): ", 0, 100),
        "sleep_hours": get_float("Sleep hours per day (0-24): ", 0, 24),
        "participation": get_float("Participation level (0-10): ", 0, 10),
    }

    label, probabilities = predict_from_values(model_path, values)
    print(f"\nPredicted performance category: {label}")
    if probabilities:
        print("Class probabilities:")
        for class_name, probability in probabilities.items():
            print(f"  {class_name}: {probability:.2%}")


def main():
    while True:
        print("\n" + "=" * 48)
        print(" STUDENT PERFORMANCE PREDICTION SYSTEM")
        print("=" * 48)
        print("1. Train models")
        print("2. Evaluate model")
        print("3. Predict student performance")
        print("4. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            train_flow()
        elif choice == "2":
            evaluation_flow()
        elif choice == "3":
            prediction_flow()
        elif choice == "4":
            print("Goodbye.")
            break
        else:
            print("Invalid choice. Select 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()

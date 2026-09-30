# Student Performance Prediction and Academic Risk Analysis

A command-line machine learning project for the **Fundamentals of AI/ML** course. The system trains classification models using student-related features and predicts a broad performance category: `Low`, `Average`, or `Good`.

> **Educational notice:** The included dataset is synthetic and intended for demonstration only. This project must not be used for high-stakes academic decisions.

## Features

- Load and validate a CSV dataset
- Preprocess numerical features
- Train Logistic Regression, Decision Tree, and Random Forest models
- Compare model metrics
- Save the selected model
- Evaluate the selected model
- Predict a new student's performance from terminal input
- Generate a confusion matrix image
- Run automated tests with pytest

## Project Structure

```text
student-performance-ml/
├── README.md
├── statement.md
├── requirements.txt
├── main.py
├── data/
│   └── student_performance.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── prediction.py
│   └── utils.py
├── models/
├── outputs/
├── tests/
└── docs/
```

## Requirements

- Python 3.10 or later recommended
- pip
- Windows, Linux, or macOS terminal

## Installation

### 1. Clone or download the repository

```bash
git clone https://github.com/YOUR-USERNAME/student-performance-ml.git
cd student-performance-ml
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
venv\Scripts\activate.bat
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Project

Start the application:

```bash
python main.py
```

Menu options:

1. Train models
2. Evaluate the saved model
3. Predict a new student's performance
4. Exit

The first option should be run before evaluation or prediction.

## Run Tests

```bash
pytest -q
```

## Dataset

The included dataset is synthetic and contains:

- `study_hours`
- `attendance`
- `previous_score`
- `assignment_completion`
- `sleep_hours`
- `participation`
- `performance`

## Example Prediction Inputs

These are example input values only:

- Study hours: `6`
- Attendance: `85`
- Previous score: `72`
- Assignment completion: `90`
- Sleep hours: `7`
- Participation: `8`

The output depends on the model trained from the dataset.

## Limitations

- The dataset is synthetic.
- The model may contain bias from the generated data.
- Performance categories are broad and do not represent actual academic ability.
- The system does not explain causation.
- The prediction is not a substitute for teacher or institutional judgment.

## Originality and Attribution

Before submission, review every file, modify the project based on your own understanding, run the tests, add your own screenshots and diagrams, and document any external references or datasets used.

# System Architecture

```text
+-----------------------+
| Command-Line Interface|
+-----------+-----------+
            |
            v
+-----------------------+
| Data Loader & Validator|
+-----------+-----------+
            |
            v
+-----------------------+
| Preprocessing Pipeline |
+-----------+-----------+
            |
            v
+-----------------------+
| Model Training         |
| LR / DT / RandomForest |
+-----------+-----------+
            |
            v
+-----------------------+
| Evaluation             |
| Accuracy / F1 / Matrix |
+-----------+-----------+
            |
            v
+-----------------------+
| Saved Model & Prediction|
+-----------------------+
```

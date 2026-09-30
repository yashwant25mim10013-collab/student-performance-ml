def print_metrics(model_name, metrics):
    print(f"\n{model_name}")
    print("-" * len(model_name))
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted):    {metrics['recall_weighted']:.4f}")
    print(f"F1-score (weighted):  {metrics['f1_weighted']:.4f}")

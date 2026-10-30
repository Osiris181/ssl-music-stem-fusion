"""Task adapters for losses, validation selection, and evaluation metrics.

Only multiclass genre classification is implemented initially. Other tasks
can provide the same interface without changing the shared training loop.
"""

from __future__ import annotations

import math
from typing import Any


class GenreMetrics:
    name = "genre_classification"
    selection_rule = "validation accuracy_micro + roc_auc_macro + average_precision_macro"
    conventions = {
        "loss": "cross_entropy", "accuracy": "micro", "f1": "macro",
        "roc_auc": "macro_one_vs_rest", "pr_auc": "macro_average_precision",
        "prediction": "argmax", "checkpoint_ties": "earliest_epoch",
        "absent_classes": "error", "zero_division_precision_recall_f1": 0,
    }

    def __init__(self, number_of_classes: int) -> None:
        import torch

        if number_of_classes < 2:
            raise ValueError("At least two classes are required.")
        self.number_of_classes = number_of_classes
        self.loss = torch.nn.CrossEntropyLoss()

    def evaluate(self, logits: Any, labels: Any, *, full: bool = True) -> dict:
        import torch
        from torchmetrics.functional.classification import (
            multiclass_accuracy, multiclass_auroc, multiclass_average_precision,
            multiclass_confusion_matrix, multiclass_f1_score, multiclass_precision, multiclass_recall,
        )

        values = torch.as_tensor(logits, dtype=torch.float32, device="cpu")
        targets = torch.as_tensor(labels, dtype=torch.long, device="cpu")
        classes = self.number_of_classes
        if values.shape != (len(targets), classes) or not torch.isfinite(values).all():
            raise ValueError("Predictions must be finite [tracks, classes] logits.")
        if set(targets.tolist()) != set(range(classes)):
            raise ValueError("Every split must contain every class for comparable macro AUCs.")
        probabilities = values.softmax(dim=1)
        roc = multiclass_auroc(probabilities, targets, num_classes=classes, average=None, thresholds=None)
        pr = multiclass_average_precision(probabilities, targets, num_classes=classes, average=None, thresholds=None)
        result = {
            "accuracy": float(multiclass_accuracy(probabilities, targets, num_classes=classes, average="micro")),
            "roc_auc_macro": float(roc.mean()), "pr_auc_macro": float(pr.mean()),
        }
        if full:
            for name, function in (("precision_macro", multiclass_precision), ("recall_macro", multiclass_recall), ("f1_macro", multiclass_f1_score)):
                result[name] = float(function(probabilities, targets, num_classes=classes, average="macro", zero_division=0))
            result["per_class"] = [{"label": index, "roc_auc": float(roc[index]), "pr_auc": float(pr[index])} for index in range(classes)]
            result["confusion_matrix"] = multiclass_confusion_matrix(probabilities, targets, num_classes=classes).tolist()
        if not all(math.isfinite(value) for value in (*roc.tolist(), *pr.tolist(), result["accuracy"])):
            raise ValueError("Evaluation produced undefined metrics.")
        return result

    def selection_score(self, metrics: dict) -> float:
        values = [metrics[name] for name in ("accuracy", "roc_auc_macro", "pr_auc_macro")]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("Checkpoint score components must be finite and on [0, 1].")
        return sum(values)

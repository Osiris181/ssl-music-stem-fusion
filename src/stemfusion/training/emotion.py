"""Binary emotion metrics for the shared Music2Emo fitting loop."""

from __future__ import annotations

import numpy as np
import torch


class BinaryEmotionTask:
    """One sigmoid output; macro metrics average the low and high classes."""

    def __init__(self, target: str):
        if target not in {"valence", "arousal"}:
            raise ValueError("Choose valence or arousal.")
        self.target = target

    def protocol(self):
        return {"task": "binary_emotion_mid_fusion", "target": self.target,
                "output_labels": 1, "positive_class": "high", "class_order": ["low", "high"],
                "accuracy": "binary", "f1": "macro_two_class",
                "roc_auc": "macro_two_class_one_vs_rest", "pr_auc": "macro_two_class_average_precision",
                "threshold": "validation_macro_f1_global_0_to_1_step_0.01_ties_lowest"}

    @staticmethod
    def _arrays(arrays):
        probabilities, labels = np.asarray(arrays["probabilities"]), np.asarray(arrays["targets"])
        if (probabilities.ndim != 2 or probabilities.shape[1] != 1 or probabilities.shape != labels.shape
                or not len(labels) or not np.isfinite(probabilities).all()
                or not ((probabilities >= 0) & (probabilities <= 1)).all()
                or not np.isin(labels, (0, 1)).all()):
            raise ValueError("Expected aligned finite [tracks, 1] binary probabilities and targets.")
        return probabilities[:, 0], labels[:, 0].astype(bool)

    def ranking_metrics(self, probabilities, labels):
        from torchmetrics.functional.classification import binary_auroc, binary_average_precision

        values, targets = self._arrays({"probabilities": probabilities.numpy(), "targets": labels.numpy()})
        if set(targets.tolist()) != {False, True}:
            raise ValueError("Each emotion split needs both low and high examples for macro AUC.")
        values = torch.as_tensor(values)
        targets = torch.as_tensor(targets, dtype=torch.long)
        roc = [float(binary_auroc(1 - values, 1 - targets, thresholds=None)),
               float(binary_auroc(values, targets, thresholds=None))]
        pr = [float(binary_average_precision(1 - values, 1 - targets, thresholds=None)),
              float(binary_average_precision(values, targets, thresholds=None))]
        if not np.isfinite([*roc, *pr]).all():
            raise ValueError("Undefined binary emotion ranking metrics.")
        return {"roc_auc_macro": float(np.mean(roc)), "pr_auc_macro": float(np.mean(pr)),
                "per_class_roc_auc": roc, "per_class_average_precision": pr}

    def threshold_metrics(self, arrays, threshold):
        values, targets = self._arrays(arrays)
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Prediction threshold must be in [0, 1].")
        predictions = values >= threshold
        tp = np.asarray([(~predictions & ~targets).sum(), (predictions & targets).sum()])
        fp = np.asarray([(~predictions & targets).sum(), (predictions & ~targets).sum()])
        fn = fp[::-1]
        def ratio(numerator, denominator):
            return np.divide(numerator, denominator, out=np.zeros(2, dtype=float), where=denominator != 0)
        accuracy = float((predictions == targets).mean())
        return {"accuracy": accuracy, "subset_accuracy": accuracy,
                "precision_macro": float(ratio(tp, tp + fp).mean()),
                "recall_macro": float(ratio(tp, tp + fn).mean()),
                "f1_macro": float(ratio(2 * tp, 2 * tp + fp + fn).mean())}

    def select_threshold(self, validation_arrays):
        self._arrays(validation_arrays)
        grid = np.linspace(0.0, 1.0, 101)
        scores = [self.threshold_metrics(validation_arrays, threshold)["f1_macro"] for threshold in grid]
        return float(grid[int(np.argmax(scores))])

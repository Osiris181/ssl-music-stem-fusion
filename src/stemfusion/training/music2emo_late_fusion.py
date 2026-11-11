"""Unweighted multilabel ensembles from independently selected branch logits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from stemfusion.training.music2emo import _save_predictions, select_threshold, threshold_metrics, weighted_bce
from stemfusion.training.reporting import file_checksum, fingerprint, finish_artifacts, reusable_artifacts, write_json


def member_identity(branch):
    directory = Path(branch["output_directory"])
    marker = json.loads((directory / "run_manifest.json").read_text())
    if marker.get("signature") != branch["signature"]:
        raise ValueError("Branch completion signature differs from its recorded result.")
    for name in ("result.json", "predictions_validation.npz", "predictions_test.npz"):
        if marker.get("files", {}).get(name) != file_checksum(directory / name):
            raise ValueError(f"Branch artifact is missing or changed: {directory / name}")
    return {"branch_id": branch["branch_id"], "source": branch["source"], "index": branch["index"],
            "actual_training_seed": branch["actual_training_seed"], "outer_seed": branch["outer_seed"],
            "signature": branch["signature"], "run_manifest_sha256": file_checksum(directory / "run_manifest.json"),
            "output_directory": str(directory), "checkpoint": branch["checkpoint"],
            "best_epoch": branch["best_epoch"], "validation_loss": branch["validation"]["loss"],
            "trainable_parameters": branch["trainable_parameters"]}


def sum_predictions(branches, split):
    ids = targets = logits = None
    for branch in branches:
        with np.load(Path(branch["output_directory"]) / f"predictions_{split}.npz", allow_pickle=False) as saved:
            current = saved["logits"]
            if ids is None:
                ids, targets = saved["track_ids"].copy(), saved["targets"].copy()
                logits = np.zeros_like(current, dtype=np.float32)
            if not np.array_equal(ids, saved["track_ids"]) or not np.array_equal(targets, saved["targets"]):
                raise ValueError("All branches require identical track IDs, ordering, and targets.")
            if current.shape != (len(ids), 56) or current.shape != logits.shape or not np.isfinite(current).all():
                raise ValueError("Expected finite source-aligned 56-label branch logits.")
            logits += current
    if logits is None or not np.isfinite(logits).all():
        raise ValueError("No valid finite ensemble logits.")
    return {"track_ids": ids, "targets": targets, "logits": logits,
            "probabilities": torch.from_numpy(logits).sigmoid().numpy()}


def score_predictions(arrays, prevalence, threshold):
    from torchmetrics.functional.classification import multilabel_auroc, multilabel_average_precision

    probabilities = torch.from_numpy(arrays["probabilities"])
    targets = torch.from_numpy(arrays["targets"])
    roc = multilabel_auroc(probabilities, targets.int(), num_labels=56, average=None, thresholds=None)
    pr = multilabel_average_precision(probabilities, targets.int(), num_labels=56, average=None, thresholds=None)
    if not torch.isfinite(roc).all() or not torch.isfinite(pr).all():
        raise ValueError("Every tag requires positive and negative examples for macro AUC/AP.")
    loss = weighted_bce(prevalence, "cpu")(torch.from_numpy(arrays["logits"]), targets.float())
    return {"loss": float(loss), "roc_auc_macro": float(roc.mean()), "pr_auc_macro": float(pr.mean()),
            "per_tag_roc_auc": roc.tolist(), "per_tag_average_precision": pr.tolist(),
            **threshold_metrics(arrays, threshold)}


def assemble_ensemble(branches, *, condition, outer_seed, output_directory, prevalence,
                      implementation, resume=True, overwrite=False):
    if len(branches) not in (4, 5, 6):
        raise ValueError("Use four, five, or six independent branches.")
    if len({branch["actual_training_seed"] for branch in branches}) != len(branches):
        raise ValueError("Ensemble branches require distinct actual training seeds.")
    if len({str(Path(branch["checkpoint"]).resolve()) for branch in branches}) != len(branches):
        raise ValueError("An ensemble cannot repeat a single trained checkpoint.")
    if any(branch["outer_seed"] != outer_seed for branch in branches):
        raise ValueError("Branches must belong to the same outer seed.")
    if len({branch["cohort_fingerprint"] for branch in branches}) != 1:
        raise ValueError("Ensemble branches must share the same track/label cohort.")
    members = [member_identity(branch) for branch in branches]
    protocol = {"aggregation": "sum_logits_then_sigmoid", "branch_selection": "minimum_validation_weighted_bce",
                "ensemble_weight_tuning": False, "ensemble_checkpoint_selection": None,
                "threshold": "validation_macro_f1_global_0_to_1_step_0.01_ties_lowest"}
    signature = fingerprint({"condition": condition, "outer_seed": outer_seed, "members": members,
                             "protocol": protocol, "implementation": implementation,
                             "cohort_fingerprint": branches[0]["cohort_fingerprint"]})
    if output_directory.exists() and not overwrite:
        if resume and reusable_artifacts(output_directory, signature):
            return json.loads((output_directory / "result.json").read_text())
        raise ValueError(f"Incomplete ensemble outputs: {output_directory}; use --overwrite.")
    arrays = {split: sum_predictions(branches, split) for split in ("validation", "test")}
    threshold = select_threshold(arrays["validation"])
    result = {"seed": outer_seed, "condition": condition, "signature": signature, "protocol": protocol,
              "members": members, "threshold": threshold, "number_of_branches": len(branches),
              "trainable_parameters": sum(branch["trainable_parameters"] for branch in branches),
              "data_fingerprint": branches[0]["cohort_fingerprint"],
              "cohort_fingerprint": branches[0]["cohort_fingerprint"], "implementation_sha256": implementation,
              "output_directory": str(output_directory)}
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "run_manifest.json").unlink(missing_ok=True)
    for split in ("validation", "test"):
        result[split] = score_predictions(arrays[split], prevalence, threshold)
        arrays[split]["threshold"] = np.asarray(threshold)
        arrays[split]["predictions"] = (arrays[split]["probabilities"] >= threshold).astype(np.uint8)
        _save_predictions(output_directory / f"predictions_{split}.npz", arrays[split])
    write_json(output_directory / "result.json", result)
    finish_artifacts(output_directory, signature, ["result.json", "predictions_validation.npz", "predictions_test.npz"])
    return result

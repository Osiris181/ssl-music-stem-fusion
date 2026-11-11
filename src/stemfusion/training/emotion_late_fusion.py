"""Binary-emotion ensembles using the Music2Emo branch artifact contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from stemfusion.training.emotion import BinaryEmotionTask
from stemfusion.training.music2emo import _save_predictions, weighted_bce
from stemfusion.training.music2emo_late_fusion import member_identity
from stemfusion.training.reporting import file_checksum, fingerprint, finish_artifacts, reusable_artifacts, write_json


class BinaryEmotionLateTask(BinaryEmotionTask):
    """Identical binary metrics with the late-fusion task identity."""

    def protocol(self):
        return {**super().protocol(), "task": "binary_emotion_late_fusion"}


def sum_predictions(branches, split):
    """Use the same float32 logit sum as Jamendo, with one binary output."""
    ids = targets = logits = None
    for branch in branches:
        with np.load(Path(branch["output_directory"]) / f"predictions_{split}.npz", allow_pickle=False) as saved:
            current = saved["logits"]
            if ids is None:
                ids, targets = saved["track_ids"].copy(), saved["targets"].copy()
                logits = np.zeros_like(current, dtype=np.float32)
            if not np.array_equal(ids, saved["track_ids"]) or not np.array_equal(targets, saved["targets"]):
                raise ValueError("All branches require identical track IDs, ordering, and targets.")
            if current.shape != (len(ids), 1) or current.shape != logits.shape or not np.isfinite(current).all():
                raise ValueError("Expected finite source-aligned one-output binary branch logits.")
            logits += current
    if logits is None or not np.isfinite(logits).all():
        raise ValueError("No valid finite ensemble logits.")
    return {"track_ids": ids, "targets": targets, "logits": logits,
            "probabilities": torch.from_numpy(logits).sigmoid().numpy()}


def score_predictions(arrays, prevalence, threshold, task):
    probabilities = torch.from_numpy(arrays["probabilities"])
    targets = torch.from_numpy(arrays["targets"])
    ranking = task.ranking_metrics(probabilities, targets)
    loss = weighted_bce(prevalence, "cpu")(torch.from_numpy(arrays["logits"]), targets.float())
    return {"loss": float(loss), **ranking, **task.threshold_metrics(arrays, threshold)}


def assemble_ensemble(branches, *, condition, outer_seed, output_directory, prevalence,
                      target, implementation, resume=True, overwrite=False):
    """Select each branch independently; fit no ensemble weights or checkpoints."""
    task = BinaryEmotionLateTask(target)
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
    if any(branch["protocol"].get("target") != target for branch in branches):
        raise ValueError("Ensemble branches must use the same binary emotion target.")
    members = [member_identity(branch) for branch in branches]
    # Verify the fitted model as well as the prediction artifacts on every reuse.
    for branch, member in zip(branches, members, strict=True):
        directory = Path(branch["output_directory"])
        marker = json.loads((directory / "run_manifest.json").read_text())
        checkpoint = Path(branch["checkpoint"])
        if checkpoint.resolve() != (directory / "best.pt").resolve():
            raise ValueError("A branch checkpoint must be its recorded best.pt.")
        checksum = file_checksum(checkpoint)
        if marker.get("files", {}).get("best.pt") != checksum:
            raise ValueError(f"Branch checkpoint is missing or changed: {checkpoint}")
        member["checkpoint_sha256"] = checksum
    protocol = {**task.protocol(), "task": "binary_emotion_late_fusion",
                "aggregation": "sum_logits_then_sigmoid", "branch_selection": "minimum_validation_weighted_bce",
                "ensemble_weight_tuning": False, "ensemble_checkpoint_selection": None}
    signature = fingerprint({"condition": condition, "outer_seed": outer_seed, "members": members,
                             "protocol": protocol, "implementation": implementation,
                             "training_prevalence": np.asarray(prevalence).tolist(),
                             "cohort_fingerprint": branches[0]["cohort_fingerprint"]})
    if output_directory.exists() and not overwrite:
        if resume and reusable_artifacts(output_directory, signature):
            return json.loads((output_directory / "result.json").read_text())
        raise ValueError(f"Incomplete ensemble outputs: {output_directory}; use --overwrite.")
    arrays = {split: sum_predictions(branches, split) for split in ("validation", "test")}
    threshold = task.select_threshold(arrays["validation"])
    result = {"seed": outer_seed, "target": target, "condition": condition, "signature": signature,
              "protocol": protocol, "members": members, "threshold": threshold, "number_of_branches": len(branches),
              "trainable_parameters": sum(branch["trainable_parameters"] for branch in branches),
              "data_fingerprint": branches[0]["cohort_fingerprint"],
              "cohort_fingerprint": branches[0]["cohort_fingerprint"], "implementation_sha256": implementation,
              "output_directory": str(output_directory)}
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "run_manifest.json").unlink(missing_ok=True)
    for split in ("validation", "test"):
        result[split] = score_predictions(arrays[split], prevalence, threshold, task)
        arrays[split]["threshold"] = np.asarray(threshold)
        arrays[split]["predictions"] = (arrays[split]["probabilities"] >= threshold).astype(np.uint8)
        _save_predictions(output_directory / f"predictions_{split}.npz", arrays[split])
    write_json(output_directory / "result.json", result)
    finish_artifacts(output_directory, signature, ["result.json", "predictions_validation.npz", "predictions_test.npz"])
    return result

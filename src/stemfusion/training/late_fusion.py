"""Independently trained branches and equal-logit-sum ensembles."""

from __future__ import annotations

import hashlib
from pathlib import Path

from stemfusion.training.mid_fusion import train_mid_condition
from stemfusion.training.reporting import file_checksum, fingerprint, finish_artifacts, write_json, write_predictions
from stemfusion.training.trainer import prepare_output

AGGREGATION = "sum_logits_then_softmax"


def branch_seed(outer_seed: int, index: int) -> int:
    """Pair the same branch position across mixture and stem ensembles."""
    if index == 0:
        return outer_seed
    value = hashlib.sha256(f"stemfusion-phase2:{outer_seed}:{index}".encode()).digest()
    return int.from_bytes(value[:4], "big")


def train_branch(*, data, source: str, index: int, outer_seed: int, root: Path, settings,
                 task, provenance: dict, resume: bool = False, overwrite: bool = False) -> dict:
    branch_id = f"mixture_{index + 1}" if source == "mixture" else source
    return train_mid_condition(data=data, sources=(source,), settings=settings,
                               seed=branch_seed(outer_seed, index),
                               output_directory=root / "branches" / f"seed_{outer_seed}" / branch_id,
                               provenance=provenance, task=task, resume=resume, overwrite=overwrite)


def assemble_ensemble(*, branches: list[dict], output_directory: Path, task, outer_seed: int,
                      condition: str, resume: bool = False, overwrite: bool = False) -> dict:
    """Combine selected checkpoints with no ensemble-specific selection."""
    import json
    import numpy as np

    if len(branches) < 2:
        raise ValueError("An ensemble requires at least two independently trained branches.")
    if len({branch["seed"] for branch in branches}) != len(branches):
        raise ValueError("Branches within an ensemble must have distinct seeds.")
    if len({branch["data_fingerprint"] for branch in branches}) != 1:
        raise ValueError("Ensemble branches must use identical dataset inputs.")
    members = [{"checkpoint_path": branch["checkpoint_path"], "seed": branch["seed"],
                "best_epoch": branch["best_epoch"], "best_validation": branch["best_validation"],
                "model": branch["model"], "trainable_parameters": branch["trainable_parameters"],
                "run_manifest_sha256": file_checksum(Path(branch["output_directory"]) / "run_manifest.json")}
               for branch in branches]
    resolved = {"aggregation": AGGREGATION, "branch_selection": task.selection_rule,
                "condition": condition, "outer_seed": outer_seed, "members": members,
                "data_fingerprint": branches[0]["data_fingerprint"], "ensemble_selection": None}
    signature = fingerprint(resolved)
    if prepare_output(output_directory, signature, resume=resume, overwrite=overwrite):
        return json.loads((output_directory / "metrics.json").read_text())
    write_json(output_directory / "ensemble.json", resolved)
    result = {**resolved, "seed": outer_seed, "trainable_parameters": sum(branch["trainable_parameters"] for branch in branches),
              "output_directory": str(output_directory)}
    for split, filename in (("validation", "validation_predictions.npz"), ("test", "predictions.npz")):
        ids = targets = summed = None
        for branch in branches:
            with np.load(Path(branch["output_directory"]) / filename, allow_pickle=False) as saved:
                if ids is None:
                    ids, targets = saved["track_ids"].copy(), saved["targets"].copy()
                    summed = np.zeros_like(saved["logits"], dtype=np.float32)
                if not np.array_equal(ids, saved["track_ids"]) or not np.array_equal(targets, saved["targets"]):
                    raise ValueError("Branch predictions must have identical track IDs, ordering, and targets.")
                if saved["logits"].shape != summed.shape or not np.isfinite(saved["logits"]).all():
                    raise ValueError("Branch prediction logits are incompatible or non-finite.")
                summed += saved["logits"]
        write_predictions(output_directory / filename, track_ids=ids, targets=targets, logits=summed)
        result[split] = task.evaluate(summed, targets)
    write_json(output_directory / "metrics.json", result)
    finish_artifacts(output_directory, signature, ["ensemble.json", "metrics.json", "validation_predictions.npz", "predictions.npz"])
    return result

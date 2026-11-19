"""Read a completed genre mixture reference without fitting or changing files."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from stemfusion.training.mid_fusion import model_parameter_count
from stemfusion.training.reporting import fingerprint, reusable_artifacts
from stemfusion.training.trainer import PROTOCOL_VERSION, implementation_fingerprint


def load_reference_branch(config, data, outer_seed, task):
    """Require the saved M1 to match this branch protocol and retained cohort."""
    import numpy as np

    directory = config.reference_output_directory / "branches" / f"seed_{outer_seed}" / "mixture_1"
    marker_path = directory / "run_manifest.json"
    if not marker_path.is_file():
        raise ValueError(f"Completed CNN mixture-only reference required; no reference training will be started: {directory}")
    marker = json.loads(marker_path.read_text())
    required = {"checkpoint.pt", "resolved_config.json", "environment.json", "metrics.json",
                "validation_predictions.npz", "predictions.npz"}
    if not required <= marker.get("files", {}).keys() or not marker.get("signature"):
        raise ValueError(f"Reference completion record is missing required artifacts: {directory}")
    reusable_artifacts(directory, marker["signature"])
    resolved = json.loads((directory / "resolved_config.json").read_text())
    result = json.loads((directory / "metrics.json").read_text())
    model = {"architecture": "deep_mlp", "sources": ["mixture"], "input_dimension": data.dimension,
             "hidden_dimensions": list(config.settings.hidden_dimensions), "number_of_classes": data.number_of_classes}
    counts = {split: len(values.track_ids) for split, values in data.splits.items()}
    expected = {
        "protocol_version": PROTOCOL_VERSION, "implementation_sha256": implementation_fingerprint(),
        "settings": asdict(config.settings), "seed": outer_seed, "task": task.name,
        "metric_conventions": task.conventions, "selection_rule": task.selection_rule,
        "data_fingerprint": data.fingerprint, "model": model, "label_mapping": data.label_mapping,
        "output_directory": str(directory), "split_examples": counts,
        "optimizer": {"name": "Adam", "lr": config.settings.learning_rate, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.0, "amsgrad": False, "foreach": False, "fused": False},
        "scheduler": None, "early_stopping": None, "deterministic_algorithms": True,
        "feature_normalization": None, "silent_vector_policy": "retain",
    }
    for key, value in expected.items():
        # JSON represents tuples as arrays; compare their serialized identities.
        if key not in resolved or fingerprint(resolved[key]) != fingerprint(value):
            raise ValueError(f"CNN mixture reference has incompatible {key}: {directory}")
    parameters = model_parameter_count(data.dimension, data.number_of_classes, config.settings.hidden_dimensions)
    for key in ("seed", "data_fingerprint", "model", "label_mapping", "output_directory", "split_examples"):
        if fingerprint(result.get(key)) != fingerprint(expected[key]):
            raise ValueError(f"CNN reference result and requested {key} disagree: {directory}")
    if resolved.get("trainable_parameters") != parameters or result.get("trainable_parameters") != parameters:
        raise ValueError(f"CNN reference parameter count differs from its classifier: {directory}")
    if Path(result.get("checkpoint_path", "")).resolve() != (directory / "checkpoint.pt").resolve():
        raise ValueError(f"CNN reference must use its recorded selected checkpoint: {directory}")
    for split, filename in (("validation", "validation_predictions.npz"), ("test", "predictions.npz")):
        values = data.splits[split]
        with np.load(directory / filename, allow_pickle=False) as saved:
            if not np.array_equal(saved["track_ids"], values.track_ids) or not np.array_equal(saved["targets"], values.labels):
                raise ValueError(f"CNN reference prediction IDs/order or labels differ: {directory / filename}")
            if saved["logits"].shape != (len(values.track_ids), data.number_of_classes) or not np.isfinite(saved["logits"]).all():
                raise ValueError(f"Invalid CNN reference logits: {directory / filename}")
    return result

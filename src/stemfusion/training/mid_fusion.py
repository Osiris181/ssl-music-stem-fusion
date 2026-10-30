"""Shared concatenation workflow and mixture-only capacity controls."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from stemfusion.training.trainer import TrainingSettings, count_parameters, fit_classifier


def model_parameter_count(input_dimension: int, classes: int, hidden: tuple[int, int]) -> int:
    import torch
    from stemfusion.models.mlp import DeepMLPClassifier

    # Count the actual architecture without allocating model weights.
    with torch.device("meta"):
        model = DeepMLPClassifier(input_dimension, classes, hidden)
    return count_parameters(model)


def match_capacity(dimension: int, classes: int, sources: int, hidden: tuple[int, int]) -> dict:
    """Find the closest integer first-hidden width using actual model counts."""
    target = model_parameter_count(dimension * sources, classes, hidden)
    intercept = model_parameter_count(dimension, classes, (1, hidden[1]))
    slope = model_parameter_count(dimension, classes, (2, hidden[1])) - intercept
    approximate = max(1, (target - intercept) // slope + 1)
    candidates = range(max(1, approximate - 1), approximate + 3)
    width = min(candidates, key=lambda candidate: (abs(model_parameter_count(dimension, classes, (candidate, hidden[1])) - target), candidate))
    matched = model_parameter_count(dimension, classes, (width, hidden[1]))
    return {"hidden_dimensions": [width, hidden[1]], "fusion_parameters": target,
            "matched_parameters": matched, "parameter_difference": matched - target,
            "relative_parameter_difference": (matched - target) / target}


def train_mid_condition(*, data, sources: tuple[str, ...], settings: TrainingSettings, seed: int,
                        output_directory: Path, provenance: dict, task, matched_sources: int | None = None,
                        resume: bool = False, overwrite: bool = False) -> dict:
    from stemfusion.models.mlp import DeepMLPClassifier

    capacity = None
    if matched_sources is not None:
        if sources != ("mixture",):
            raise ValueError("Capacity-matched controls receive mixture features only.")
        capacity = match_capacity(data.dimension, data.number_of_classes, matched_sources, settings.hidden_dimensions)
        settings = replace(settings, hidden_dimensions=tuple(capacity["hidden_dimensions"]))
    input_dimension = data.dimension * len(sources)
    description = {"architecture": "deep_mlp", "sources": list(sources), "input_dimension": input_dimension,
                   "hidden_dimensions": list(settings.hidden_dimensions), "number_of_classes": data.number_of_classes}
    result = fit_classifier(
        arrays={split: values.concatenate(sources) for split, values in data.splits.items()},
        labels={split: values.labels for split, values in data.splits.items()},
        track_ids={split: values.track_ids for split, values in data.splits.items()},
        model_factory=lambda: DeepMLPClassifier(input_dimension, data.number_of_classes, settings.hidden_dimensions, settings.dropout),
        task=task, settings=settings, seed=seed, output_directory=output_directory,
        data_fingerprint=data.fingerprint, model_description=description, provenance=provenance,
        label_mapping=data.label_mapping, resume=resume, overwrite=overwrite,
    )
    if capacity is not None:
        result = {**result, "capacity_matching": capacity}
    return result

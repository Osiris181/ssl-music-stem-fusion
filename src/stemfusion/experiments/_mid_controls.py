"""Repeated-mixture input controls using the unchanged shared fitting backend.

Input construction is separate from the fit backend so existing standard/fusion
fits keep their verified implementation identity. New controls explicitly include
this module's checksum in their model description and checkpoint signature.
"""

from __future__ import annotations

from pathlib import Path

from stemfusion.training.mid_fusion import model_parameter_count
from stemfusion.training.reporting import file_checksum
from stemfusion.training.trainer import fit_classifier


def repeated_mixture_spec(dimension: int, classes: int, copies: int, hidden: tuple[int, int]) -> dict:
    """Match fusion input width and parameter count without changing hidden widths."""
    if type(copies) is not int or copies not in (4, 5, 6):
        raise ValueError("Repeated-mixture controls require 4, 5, or 6 identical copies.")
    parameters = model_parameter_count(dimension * copies, classes, hidden)
    return {"method": "repeat_identical_mixture_embedding", "copies": copies,
            "input_dimension": dimension * copies, "dimension_per_copy": dimension,
            "hidden_dimensions": list(hidden), "fusion_parameters": parameters,
            "matched_parameters": parameters, "parameter_difference": 0,
            "relative_parameter_difference": 0.0}


def train_repeated_mixture_control(*, data, copies: int, settings, seed: int,
                                   output_directory: Path, provenance: dict, task,
                                   resume: bool = False, overwrite: bool = False) -> dict:
    """Concatenate exact copies of each track's mixture vector, then fit one MLP."""
    import numpy as np
    from stemfusion.models.mlp import DeepMLPClassifier

    matching = repeated_mixture_spec(data.dimension, data.number_of_classes, copies, settings.hidden_dimensions)
    arrays = {}
    for split, values in data.splits.items():
        mixture = values.features["mixture"]
        if mixture.ndim != 2 or mixture.shape != (len(values.track_ids), data.dimension):
            raise ValueError("Mixture features must be track-aligned vectors of the configured width.")
        arrays[split] = np.concatenate([mixture] * copies, axis=1)
    description = {"architecture": "deep_mlp", "sources": ["mixture"] * copies,
                   "input_dimension": matching["input_dimension"],
                   "hidden_dimensions": list(settings.hidden_dimensions), "number_of_classes": data.number_of_classes,
                   "input_transform": {"name": matching["method"], "copies": copies,
                                       "implementation_sha256": file_checksum(Path(__file__))}}
    result = fit_classifier(
        arrays=arrays, labels={split: values.labels for split, values in data.splits.items()},
        track_ids={split: values.track_ids for split, values in data.splits.items()},
        model_factory=lambda: DeepMLPClassifier(matching["input_dimension"], data.number_of_classes,
                                                settings.hidden_dimensions, settings.dropout),
        task=task, settings=settings, seed=seed, output_directory=output_directory,
        data_fingerprint=data.fingerprint, model_description=description, provenance=provenance,
        label_mapping=data.label_mapping, resume=resume, overwrite=overwrite)
    return {**result, "input_matching": matching}

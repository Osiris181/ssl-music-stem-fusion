"""Collect standalone FMA stem scores from completed late-fusion branches.

This module reads saved artifacts only. It never trains a classifier,
selects another checkpoint, or loads audio or feature caches.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from stemfusion.data.fma_features import SOURCES, inspect_manifest
from stemfusion.training.late_fusion import branch_seed
from stemfusion.training.reporting import file_checksum, fingerprint, reusable_artifacts, summarize_results


def collect_stem_results(config) -> dict:
    """Verify the five source branches and shared mixture reference per seed."""
    import numpy as np

    records, mapping = inspect_manifest(config.manifest_path, config.number_of_classes)
    expected = {}
    for split in ("train", "validation", "test"):
        selected = [record for record in records if record.split == split]
        expected[split] = (np.asarray([record.track_id for record in selected]),
                           np.asarray([record.label for record in selected], dtype=np.int64))
    counts = {split: len(values[0]) for split, values in expected.items()}
    results = {f"{source}_only": {} for source in SOURCES[1:]}
    mixture = {}
    baseline = None
    branch_artifacts = {}
    common_fields = ("protocol_version", "implementation_sha256", "settings", "task", "metric_conventions",
                     "selection_rule", "data_fingerprint", "label_mapping", "optimizer", "scheduler",
                     "early_stopping", "deterministic_algorithms", "feature_normalization", "silent_vector_policy", "split_examples")
    for outer_seed in config.seeds:
        branch_artifacts[str(outer_seed)] = {}
        for index, source in enumerate(SOURCES):
            branch = "mixture_1" if source == "mixture" else source
            directory = config.output_directory / "branches" / f"seed_{outer_seed}" / branch
            marker_path = directory / "run_manifest.json"
            if not marker_path.is_file():
                raise ValueError(f"Completed branch required for reporting; no training will be started: {directory}")
            marker = json.loads(marker_path.read_text())
            required = {"metrics.json", "resolved_config.json", "checkpoint.pt", "validation_predictions.npz", "predictions.npz"}
            if not required <= marker.get("files", {}).keys():
                raise ValueError(f"Branch completion record is missing required artifacts: {directory}")
            reusable_artifacts(directory, marker["signature"])
            record = json.loads((directory / "metrics.json").read_text())
            resolved = json.loads((directory / "resolved_config.json").read_text())
            model = record["model"]
            if model["sources"] != [source] or model["input_dimension"] != config.dimension or model["number_of_classes"] != config.number_of_classes:
                raise ValueError(f"Unexpected source, embedding width, or class count: {directory}")
            if record["seed"] != branch_seed(outer_seed, index) or resolved["seed"] != record["seed"]:
                raise ValueError(f"Branch seed does not follow the completed FMA seed convention: {directory}")
            if record["label_mapping"] != mapping or record["split_examples"] != counts:
                raise ValueError(f"Branch labels or split counts differ from the manifest: {directory}")
            common = {field: resolved[field] for field in common_fields}
            if baseline is None:
                baseline = common
            if fingerprint(common) != fingerprint(baseline):
                raise ValueError(f"Standalone branches must share data and training/evaluation settings: {directory}")
            if record["data_fingerprint"] != resolved["data_fingerprint"] or model != resolved["model"]:
                raise ValueError(f"Branch result and resolved configuration disagree: {directory}")
            for split, filename in (("validation", "validation_predictions.npz"), ("test", "predictions.npz")):
                ids, labels = expected[split]
                with np.load(directory / filename, allow_pickle=False) as saved:
                    if not np.array_equal(saved["track_ids"], ids) or not np.array_equal(saved["targets"], labels):
                        raise ValueError(f"Branch prediction IDs or labels differ from the manifest: {directory / filename}")
                    if saved["logits"].shape != (len(ids), config.number_of_classes) or not np.isfinite(saved["logits"]).all():
                        raise ValueError(f"Invalid saved branch logits: {directory / filename}")
            enriched = {**record, "outer_seed": outer_seed, "branch_seed": record["seed"]}
            if source == "mixture":
                mixture[outer_seed] = enriched
            else:
                results[f"{source}_only"][outer_seed] = enriched
            branch_artifacts[str(outer_seed)][source] = {
                "directory": str(directory), "run_manifest_sha256": file_checksum(marker_path),
                "metrics_sha256": marker["files"]["metrics.json"],
                "selected_checkpoint_sha256": marker["files"]["checkpoint.pt"],
            }
    summary = summarize_results(results, config.seeds, {})
    summary.update({
        "family": "stem_only", "dataset": "FMA-Medium", "analysis_role": "supporting",
        "number_of_classes": config.number_of_classes, "feature_dimension": config.dimension,
        "split_counts": counts, "training_settings": baseline["settings"],
        "data_fingerprint": baseline["data_fingerprint"], "runs": results,
        "mixture_only_reference": summarize_results({"mixture_only": mixture}, config.seeds, {})["conditions"]["mixture_only"],
        "mixture_only_runs": mixture, "label_mapping": mapping,
        "manifest_path": str(config.manifest_path), "manifest_sha256": file_checksum(config.manifest_path),
        "branch_artifacts": branch_artifacts, "new_training_performed": False,
        "new_inference_performed": False, "checkpoint_reselection_performed": False,
        "metric_conventions": baseline["metric_conventions"], "selection_rule": baseline["selection_rule"],
        "interpretation": "Standalone source-classifier performance is supporting analysis; it does not establish incremental predictive value beyond the mixture.",
    })
    return summary


def stem_report_config(config):
    """Use the same naming rules without a fusion-condition filename suffix."""
    return replace(config, family="stem_only", selected_run=None)

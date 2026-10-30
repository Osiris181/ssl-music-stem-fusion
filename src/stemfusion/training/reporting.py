"""Atomic research artifacts, checksums, and paired training-run summaries."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

METRICS = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro", "pr_auc_macro")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_predictions(path: Path, *, track_ids: Any, targets: Any, logits: Any) -> None:
    import numpy as np

    values = np.asarray(logits, dtype=np.float32)
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, track_ids=np.asarray(track_ids), targets=targets,
                                logits=values, probabilities=probabilities, predictions=values.argmax(axis=1))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def finish_artifacts(directory: Path, signature: str, filenames: list[str]) -> None:
    """Publish completion only after every listed artifact is durable."""
    write_json(directory / "run_manifest.json", {
        "signature": signature,
        "files": {name: file_checksum(directory / name) for name in filenames},
    })


def reusable_artifacts(directory: Path, signature: str) -> bool:
    marker = directory / "run_manifest.json"
    if not marker.is_file():
        return False
    data = json.loads(marker.read_text())
    if data.get("signature") != signature:
        raise ValueError(f"Existing run has incompatible settings or inputs: {directory}")
    files = data.get("files", {})
    if not files:
        raise ValueError(f"Existing run has no artifact checksums: {directory}")
    for name, expected in files.items():
        path = directory / name
        if not path.is_file() or file_checksum(path) != expected:
            raise ValueError(f"Existing run artifact is missing or changed: {path}")
    return True


def summarize_results(results: dict[str, dict[int, dict]], seeds: tuple[int, ...], comparisons: dict[str, str]) -> dict:
    """Summarize only exact, complete outer-seed pairs; SD uses ddof=1."""
    def describe(values: list[float]) -> dict:
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("Summary metrics must be finite and nonempty.")
        return {"mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                "values": values}

    conditions = {}
    for name, runs in results.items():
        if set(runs) != set(seeds):
            raise ValueError(f"Incomplete seed set for {name}.")
        conditions[name] = {metric: describe([float(runs[seed]["test"][metric]) for seed in seeds]) for metric in METRICS}
    paired = {}
    for fusion, control in comparisons.items():
        if fusion not in results or control not in results:
            continue
        for seed in seeds:
            if results[fusion][seed]["data_fingerprint"] != results[control][seed]["data_fingerprint"]:
                raise ValueError("Paired results must use identical data.")
        paired[fusion] = {"control": control, "metrics": {
            metric: describe([float(results[fusion][seed]["test"][metric]) - float(results[control][seed]["test"][metric]) for seed in seeds])
            for metric in METRICS}}
    return {"seeds": list(seeds), "number_of_runs": len(seeds), "sd_ddof": 1,
            "conditions": conditions, "paired_differences": paired,
            "significance_tests_performed": False}

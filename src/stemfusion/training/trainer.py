"""Shared model fitting and validation-only checkpoint selection.

Dataset adapters supply aligned arrays. Task adapters supply the loss and
metrics. This module has no FMA filenames, source combinations, or label rules.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import random
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from stemfusion.training.reporting import (
    file_checksum, fingerprint, finish_artifacts, reusable_artifacts, write_json, write_predictions,
)

LOGGER = logging.getLogger(__name__)
PROTOCOL_VERSION = "phase2-independent-branches-v1"


class TrainingError(ValueError):
    """The training configuration, data, or output directory is invalid."""


@dataclass(frozen=True)
class TrainingSettings:
    hidden_dimensions: tuple[int, int] = (1024, 512)
    dropout: float = 0.2
    learning_rate: float = 1e-4
    batch_size: int = 128
    epochs: int = 20
    device: str = "auto"
    num_workers: int = 0

    def validate(self) -> None:
        if len(self.hidden_dimensions) != 2 or any(type(width) is not int or width <= 0 for width in self.hidden_dimensions):
            raise TrainingError("Two positive integer hidden dimensions are required.")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise TrainingError("Dropout must be finite and in [0, 1).")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise TrainingError("Learning rate must be finite and positive.")
        if self.batch_size <= 0 or self.epochs <= 0 or self.num_workers < 0:
            raise TrainingError("Batch size/epochs must be positive and workers nonnegative.")


def implementation_fingerprint() -> str:
    package = Path(__file__).resolve().parents[1]
    files = [*Path(__file__).parent.glob("*.py"), package / "models/mlp.py", package / "data/features.py", package / "data/fma_features.py"]
    return fingerprint({str(path.relative_to(package)): file_checksum(path) for path in sorted(files)})


def seed_worker(_: int) -> None:
    import numpy as np
    import torch

    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    if not 0 <= seed < 2**32:
        raise TrainingError("Seed must be in [0, 2**32).")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def prepare_output(directory: Path, signature: str, *, resume: bool, overwrite: bool) -> bool:
    if resume and overwrite:
        raise TrainingError("Choose either --resume or --overwrite.")
    if resume and reusable_artifacts(directory, signature):
        return True
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise TrainingError(f"Output already exists or is incomplete: {directory}. Use a new root or --overwrite.")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_manifest.json").unlink(missing_ok=True)
    return False


def fit_classifier(
    *, arrays: dict[str, Any], labels: dict[str, Any], track_ids: dict[str, tuple[str, ...]],
    model_factory: Callable[[], Any], task: Any, settings: TrainingSettings,
    seed: int, output_directory: Path, data_fingerprint: str,
    model_description: dict, provenance: dict, label_mapping: dict,
    resume: bool = False, overwrite: bool = False,
) -> dict:
    """Fit one model; evaluate test only after selecting its best checkpoint."""
    import numpy as np
    import torch
    import torchmetrics
    from torch.utils.data import DataLoader, TensorDataset

    settings.validate()
    set_seed(seed)
    device_name = settings.device
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA was requested but is unavailable.")
    environment = {"numpy": np.__version__, "torch": torch.__version__, "torchmetrics": torchmetrics.__version__,
                   "python": sys.version, "platform": platform.platform(), "device": str(device),
                   "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                   "cpu_threads": torch.get_num_threads(),
                   "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()}
    resolved = {
        "protocol_version": PROTOCOL_VERSION, "implementation_sha256": implementation_fingerprint(),
        "settings": asdict(settings), "seed": seed, "task": task.name,
        "metric_conventions": task.conventions, "selection_rule": task.selection_rule,
        "data_fingerprint": data_fingerprint, "model": model_description, "label_mapping": label_mapping,
        "output_directory": str(output_directory),
        "optimizer": {"name": "Adam", "lr": settings.learning_rate, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.0, "amsgrad": False, "foreach": False, "fused": False},
        "scheduler": None, "early_stopping": None, "deterministic_algorithms": True,
        "feature_normalization": None, "silent_vector_policy": "retain",
        "split_examples": {split: len(track_ids[split]) for split in track_ids},
    }
    signature = fingerprint({"resolved": resolved, "environment": environment})
    if prepare_output(output_directory, signature, resume=resume, overwrite=overwrite):
        LOGGER.info("Reusing verified run %s", output_directory)
        return json.loads((output_directory / "metrics.json").read_text())

    # Initialization happens only after seeding, independently for every fit.
    model = model_factory().to(device)
    parameters = count_parameters(model)
    resolved.update({"trainable_parameters": parameters, "config_provenance": provenance})
    write_json(output_directory / "resolved_config.json", resolved)
    write_json(output_directory / "environment.json", environment)
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(TensorDataset(torch.from_numpy(arrays[split]), torch.from_numpy(labels[split])),
                          batch_size=settings.batch_size, shuffle=split == "train", generator=generator,
                          num_workers=settings.num_workers, worker_init_fn=seed_worker, pin_memory=device.type == "cuda")
        for split in ("train", "validation", "test")
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=0.0, amsgrad=False, foreach=False, fused=False)

    def predict(split: str) -> Any:
        model.eval()
        batches = []
        with torch.inference_mode():
            for features, _ in loaders[split]:
                batches.append(model(features.to(device)).detach().cpu().numpy())
        return np.concatenate(batches)

    best_score = -math.inf
    best_epoch = 0
    best_state = None
    best_validation = None
    history = []
    with tqdm(range(1, settings.epochs + 1), desc=output_directory.name, unit="epoch", dynamic_ncols=True) as progress:
        for epoch in progress:
            model.train()
            total_loss = 0.0
            for features, targets in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                logits = model(features.to(device))
                loss = task.loss(logits, targets.to(device))
                if not torch.isfinite(loss):
                    raise TrainingError("Training loss is not finite.")
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(targets)
            validation = task.evaluate(predict("validation"), labels["validation"], full=False)
            score = task.selection_score(validation)
            history.append({"epoch": epoch, "training_loss": total_loss / len(labels["train"]),
                            "validation": validation, "selection_score": score})
            progress.set_postfix(loss=f"{history[-1]['training_loss']:.4f}", val_accuracy=f"{validation['accuracy']:.4f}", score=f"{score:.4f}")
            if score > best_score:
                best_score, best_epoch, best_validation = score, epoch, validation
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                temporary = output_directory / ".checkpoint.pt.tmp"
                torch.save({"model_state_dict": best_state, "best_epoch": best_epoch,
                            "best_validation": best_validation, "selection_score": best_score,
                            "resolved_config": resolved}, temporary)
                temporary.replace(output_directory / "checkpoint.pt")
            write_json(output_directory / "training_history.json", history)
    if best_state is None:
        raise TrainingError("No valid checkpoint was selected.")
    model.load_state_dict(best_state)
    result = {"seed": seed, "data_fingerprint": data_fingerprint, "trainable_parameters": parameters,
              "model": model_description, "best_epoch": best_epoch, "best_validation": best_validation,
              "validation_selection_score": best_score, "checkpoint_path": str(output_directory / "checkpoint.pt"),
              "output_directory": str(output_directory), "label_mapping": label_mapping,
              "split_examples": resolved["split_examples"]}
    for split, filename in (("validation", "validation_predictions.npz"), ("test", "predictions.npz")):
        logits = predict(split)
        write_predictions(output_directory / filename, track_ids=track_ids[split], targets=labels[split], logits=logits)
        result[split] = task.evaluate(logits, labels[split])
    write_json(output_directory / "metrics.json", result)
    finish_artifacts(output_directory, signature, ["checkpoint.pt", "resolved_config.json", "environment.json",
                     "training_history.json", "validation_predictions.npz", "predictions.npz", "metrics.json"])
    return result

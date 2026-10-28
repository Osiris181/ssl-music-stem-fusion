"""Train the paper's MLP on cached FMA MusicFM features."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import pickle
import platform
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from stemfusion.data.fma import FMAManifestError, FMAManifestRecord, read_manifest

LOGGER = logging.getLogger(__name__)

CANONICAL_SOURCE_ORDER = (
    "mixture",
    "guitar",
    "drums",
    "piano",
    "bass",
    "vocals",
    "other",
)
SOURCE_ALIASES = {
    "mix": "mixture",
    "original": "mixture",
    "electric_guitar": "guitar",
    "drum": "drums",
    "vocal": "vocals",
}
SPLITS = ("train", "validation", "test")
OUTPUT_FILENAMES = (
    "checkpoint.pt",
    "data_manifest_checksum.txt",
    "environment.json",
    "metrics.json",
    "predictions.npz",
    "resolved_config.json",
    "training_history.json",
)


class GenreTrainingError(ValueError):
    """Raised when genre training cannot be configured or completed."""


@dataclass(frozen=True)
class SourceFeatureDirectory:
    """An optional per-source override for a feature-cache directory."""

    source: str
    directory: Path


@dataclass(frozen=True)
class GenreTrainingConfig:
    """Configuration for the paper-aligned FMA genre classifier."""

    manifest_path: Path
    feature_directory: Path
    output_directory: Path
    sources: tuple[str, ...]
    source_feature_directories: tuple[SourceFeatureDirectory, ...] = ()
    expected_feature_dimension: int = 1024
    hidden_dimensions: tuple[int, int] = (1024, 512)
    dropout: float = 0.2
    batch_size: int = 128
    learning_rate: float = 1e-4
    epochs: int = 20
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    allow_missing_features: bool = False
    drop_silent_features: bool = False
    overwrite: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class _TrainingDependencies:
    numpy: Any
    torch: Any
    torchmetrics: Any
    data_loader: Any
    tensor_dataset: Any
    accuracy: Any
    precision: Any
    recall: Any
    f1_score: Any
    auroc: Any
    average_precision: Any
    confusion_matrix: Any


@dataclass(frozen=True)
class _PreparedSplit:
    features: Any
    labels: Any
    track_ids: tuple[str, ...]
    dropped: int


def normalize_source(source: str) -> str:
    """Convert legacy source labels to the repository's canonical labels."""

    normalized = SOURCE_ALIASES.get(source.strip().lower(), source.strip().lower())
    if normalized not in CANONICAL_SOURCE_ORDER:
        allowed = ", ".join(CANONICAL_SOURCE_ORDER)
        raise GenreTrainingError(
            f"Unsupported source {source!r}; choose from {allowed}."
        )
    return normalized


def canonicalize_sources(sources: Sequence[str]) -> tuple[str, ...]:
    """Return selected sources in the fixed fusion order used by the paper."""

    if not sources:
        raise GenreTrainingError("At least one feature source is required.")
    normalized = [normalize_source(source) for source in sources]
    if len(normalized) != len(set(normalized)):
        raise GenreTrainingError("Each feature source may be selected only once.")
    selected = set(normalized)
    return tuple(
        source for source in CANONICAL_SOURCE_ORDER if source in selected
    )


def track_id_from_cache_key(key: Any) -> str:
    """Normalize a clean track ID or a legacy audio path to six digits."""

    text = str(key)
    filename = Path(text).stem
    candidate = filename.split("_", maxsplit=1)[0]
    if not candidate.isdigit():
        raise GenreTrainingError(
            f"Could not derive a numeric FMA track ID from cache key {key!r}."
        )
    number = int(candidate)
    if number > 999_999:
        raise GenreTrainingError(f"FMA track ID is too large: {candidate!r}.")
    return f"{number:06d}"


def normalize_feature_cache(data: Any) -> dict[str, Any]:
    """Convert clean or legacy cache containers to a track-ID mapping."""

    if isinstance(data, Mapping):
        items = data.items()
    else:
        try:
            items = iter(data)
        except TypeError as error:
            raise GenreTrainingError(
                "A feature cache must be a mapping or an iterable of pairs."
            ) from error

    normalized: dict[str, Any] = {}
    for index, item in enumerate(items):
        try:
            key, feature = item[0], item[1]
        except (IndexError, KeyError, TypeError) as error:
            raise GenreTrainingError(
                f"Feature-cache entry {index} is not a key-feature pair."
            ) from error
        track_id = track_id_from_cache_key(key)
        if track_id in normalized:
            raise GenreTrainingError(
                f"Feature cache contains duplicate track ID {track_id}."
            )
        normalized[track_id] = feature
    if not normalized:
        raise GenreTrainingError("Feature cache is empty.")
    return normalized


def _source_directory_map(
    config: GenreTrainingConfig,
) -> dict[str, Path]:
    directories: dict[str, Path] = {}
    for override in config.source_feature_directories:
        source = normalize_source(override.source)
        if source in directories:
            raise GenreTrainingError(
                f"Feature directory was configured twice for {source}."
            )
        directories[source] = override.directory
    unused = set(directories) - set(config.sources)
    if unused:
        names = ", ".join(sorted(unused))
        raise GenreTrainingError(
            f"Feature-directory overrides were provided for unselected sources: "
            f"{names}."
        )
    return directories


def resolve_feature_cache_path(
    config: GenreTrainingConfig,
    source: str,
    split: str,
) -> Path:
    """Resolve a canonical cache, with explicit legacy-directory support."""

    directories = _source_directory_map(config)
    explicit_override = source in directories
    root = directories.get(source, config.feature_directory)
    canonical = root / f"fma_musicfm_{source}_{split}.pkl"
    candidates = [canonical]
    if source == "mixture":
        candidates.append(root / f"fma_musicfm_mix_{split}.pkl")
    if explicit_override or len(config.sources) == 1:
        candidates.append(root / f"fma_musicfm_{split}.pkl")
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    listed = ", ".join(str(path) for path in candidates)
    raise GenreTrainingError(
        f"No non-empty {source} cache was found for {split}. Checked: {listed}."
    )


def _validate_config(config: GenreTrainingConfig) -> GenreTrainingConfig:
    sources = canonicalize_sources(config.sources)
    if sources != config.sources:
        config = replace(config, sources=sources)
    if not config.manifest_path.is_file():
        raise GenreTrainingError(
            f"Manifest does not exist: {config.manifest_path}"
        )
    source_directories = _source_directory_map(config)
    for source, directory in source_directories.items():
        if not directory.is_dir():
            raise GenreTrainingError(
                f"Feature directory for {source} does not exist: {directory}"
            )
    sources_using_common = set(config.sources) - set(source_directories)
    if sources_using_common and not config.feature_directory.is_dir():
        raise GenreTrainingError(
            f"Feature directory does not exist: {config.feature_directory}"
        )
    if config.expected_feature_dimension <= 0:
        raise GenreTrainingError("Expected feature dimension must be positive.")
    if len(config.hidden_dimensions) != 2 or any(
        dimension <= 0 for dimension in config.hidden_dimensions
    ):
        raise GenreTrainingError(
            "Exactly two positive hidden dimensions are required."
        )
    if not 0.0 <= config.dropout < 1.0:
        raise GenreTrainingError("Dropout must be in the interval [0, 1).")
    if config.batch_size <= 0:
        raise GenreTrainingError("Batch size must be positive.")
    if config.learning_rate <= 0:
        raise GenreTrainingError("Learning rate must be positive.")
    if config.epochs <= 0:
        raise GenreTrainingError("Number of epochs must be positive.")
    if config.num_workers < 0:
        raise GenreTrainingError("Number of data-loader workers cannot be negative.")
    return config


def _read_records(path: Path) -> list[FMAManifestRecord]:
    try:
        records = read_manifest(path)
    except FMAManifestError as error:
        raise GenreTrainingError(str(error)) from error
    split_counts = Counter(record.split for record in records)
    missing_splits = [split for split in SPLITS if not split_counts[split]]
    if missing_splits:
        names = ", ".join(missing_splits)
        raise GenreTrainingError(f"Manifest has no records for splits: {names}.")

    label_to_genre: dict[int, str] = {}
    genre_to_label: dict[str, int] = {}
    for record in records:
        if record.label < 0:
            raise GenreTrainingError(
                f"Track {record.track_id} has a negative class label."
            )
        previous_genre = label_to_genre.setdefault(record.label, record.genre)
        previous_label = genre_to_label.setdefault(record.genre, record.label)
        if previous_genre != record.genre or previous_label != record.label:
            raise GenreTrainingError(
                "Manifest genre names and numeric labels are inconsistent."
            )
    labels = sorted(label_to_genre)
    if labels != list(range(len(labels))):
        raise GenreTrainingError(
            "Manifest labels must be contiguous integers beginning at zero."
        )
    if len(labels) < 2:
        raise GenreTrainingError("At least two genre classes are required.")
    return records


def _label_mapping(records: Sequence[FMAManifestRecord]) -> dict[str, int]:
    return dict(
        sorted(
            {record.genre: record.label for record in records}.items(),
            key=lambda item: item[1],
        )
    )


def _config_dict(config: GenreTrainingConfig) -> dict[str, Any]:
    return {
        "manifest_path": str(config.manifest_path),
        "feature_directory": str(config.feature_directory),
        "output_directory": str(config.output_directory),
        "sources": list(config.sources),
        "source_feature_directories": {
            item.source: str(item.directory)
            for item in config.source_feature_directories
        },
        "expected_feature_dimension": config.expected_feature_dimension,
        "hidden_dimensions": list(config.hidden_dimensions),
        "dropout": config.dropout,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "seed": config.seed,
        "device": config.device,
        "num_workers": config.num_workers,
        "allow_missing_features": config.allow_missing_features,
        "drop_silent_features": config.drop_silent_features,
        "overwrite": config.overwrite,
    }


def plan_genre_training(config: GenreTrainingConfig) -> dict[str, Any]:
    """Validate paths and describe a run without loading large caches."""

    config = _validate_config(config)
    records = _read_records(config.manifest_path)
    cache_paths = {
        source: {
            split: str(resolve_feature_cache_path(config, source, split))
            for split in SPLITS
        }
        for source in config.sources
    }
    split_counts = Counter(record.split for record in records)
    return {
        "dry_run": True,
        "sources": list(config.sources),
        "source_order": list(config.sources),
        "input_dimension": (
            config.expected_feature_dimension * len(config.sources)
        ),
        "number_of_classes": len(_label_mapping(records)),
        "manifest_split_counts": {
            split: split_counts[split] for split in SPLITS
        },
        "feature_caches": cache_paths,
        "output_directory": str(config.output_directory),
        "training": _config_dict(config),
    }


def _load_dependencies() -> _TrainingDependencies:
    hint = 'Install the project with the "training" optional dependencies.'
    try:
        import numpy
        import torch
        import torchmetrics
        from torch.utils.data import DataLoader, TensorDataset
        from torchmetrics.classification import (
            MulticlassAUROC,
            MulticlassAccuracy,
            MulticlassAveragePrecision,
            MulticlassConfusionMatrix,
            MulticlassF1Score,
            MulticlassPrecision,
            MulticlassRecall,
        )
    except Exception as error:
        raise GenreTrainingError(
            f"Could not import the training dependencies: {error}. {hint}"
        ) from error
    return _TrainingDependencies(
        numpy=numpy,
        torch=torch,
        torchmetrics=torchmetrics,
        data_loader=DataLoader,
        tensor_dataset=TensorDataset,
        accuracy=MulticlassAccuracy,
        precision=MulticlassPrecision,
        recall=MulticlassRecall,
        f1_score=MulticlassF1Score,
        auroc=MulticlassAUROC,
        average_precision=MulticlassAveragePrecision,
        confusion_matrix=MulticlassConfusionMatrix,
    )


def _load_feature_cache(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return normalize_feature_cache(pickle.load(handle))
    except GenreTrainingError:
        raise
    except Exception as error:
        raise GenreTrainingError(
            f"Could not read feature cache {path}: {error}"
        ) from error


def _feature_vector(
    feature: Any,
    source: str,
    track_id: str,
    expected_dimension: int,
    numpy: Any,
) -> Any:
    if hasattr(feature, "detach"):
        feature = feature.detach().cpu().numpy()
    try:
        vector = numpy.asarray(feature, dtype=numpy.float32)
    except Exception as error:
        raise GenreTrainingError(
            f"Could not convert {source} feature for {track_id}: {error}"
        ) from error
    if vector.ndim == 2:
        vector = vector.mean(axis=0, dtype=numpy.float32)
    if vector.shape != (expected_dimension,):
        raise GenreTrainingError(
            f"Expected {source} feature {track_id} to have shape "
            f"({expected_dimension},), received {tuple(vector.shape)}."
        )
    if not numpy.isfinite(vector).all():
        raise GenreTrainingError(
            f"Feature for {source} track {track_id} contains non-finite values."
        )
    return vector


def _prepare_split(
    config: GenreTrainingConfig,
    records: Sequence[FMAManifestRecord],
    split: str,
    dependencies: _TrainingDependencies,
) -> _PreparedSplit:
    split_records = [record for record in records if record.split == split]
    caches = {
        source: _load_feature_cache(
            resolve_feature_cache_path(config, source, split)
        )
        for source in config.sources
    }
    rows: list[Any] = []
    labels: list[int] = []
    track_ids: list[str] = []
    missing_counts: Counter[str] = Counter()
    missing_examples: list[str] = []

    for record in split_records:
        missing = [
            source
            for source in config.sources
            if record.track_id not in caches[source]
        ]
        if missing:
            missing_counts.update(missing)
            if len(missing_examples) < 10:
                missing_examples.append(
                    f"{record.track_id} ({', '.join(missing)})"
                )
            if config.allow_missing_features:
                continue
            continue
        vectors = [
            _feature_vector(
                caches[source][record.track_id],
                source,
                record.track_id,
                config.expected_feature_dimension,
                dependencies.numpy,
            )
            for source in config.sources
        ]
        if config.drop_silent_features and any(
            not dependencies.numpy.any(vector) for vector in vectors
        ):
            missing_counts.update(["silent_feature"])
            if len(missing_examples) < 10:
                missing_examples.append(f"{record.track_id} (silent feature)")
            continue
        rows.append(dependencies.numpy.concatenate(vectors))
        labels.append(record.label)
        track_ids.append(record.track_id)

    if missing_counts and not config.allow_missing_features:
        counts = ", ".join(
            f"{source}={count}"
            for source, count in sorted(missing_counts.items())
        )
        examples = "; ".join(missing_examples)
        raise GenreTrainingError(
            f"Feature caches are incomplete for {split}. Missing counts: "
            f"{counts}. Examples: {examples}. Use --allow-missing-features "
            "only to reproduce legacy runs that silently used an intersection."
        )
    if not rows:
        raise GenreTrainingError(
            f"No aligned examples remain in the {split} split."
        )
    features = dependencies.numpy.stack(rows).astype(
        dependencies.numpy.float32,
        copy=False,
    )
    label_array = dependencies.numpy.asarray(
        labels,
        dtype=dependencies.numpy.int64,
    )
    expected_input = config.expected_feature_dimension * len(config.sources)
    if features.shape[1] != expected_input:
        raise GenreTrainingError(
            f"Expected input dimension {expected_input}, received "
            f"{features.shape[1]}."
        )
    return _PreparedSplit(
        features=features,
        labels=label_array,
        track_ids=tuple(track_ids),
        dropped=len(split_records) - len(track_ids),
    )


def _validate_class_coverage(
    prepared: Mapping[str, _PreparedSplit],
    number_of_classes: int,
) -> None:
    expected = set(range(number_of_classes))
    for split, data in prepared.items():
        present = {int(label) for label in data.labels.tolist()}
        missing = sorted(expected - present)
        if missing:
            raise GenreTrainingError(
                f"The {split} split is missing genre labels {missing}; macro "
                "ROC-AUC and PR-AUC would not be comparable to the paper."
            )


def _set_seed(seed: int, dependencies: _TrainingDependencies) -> None:
    random.seed(seed)
    dependencies.numpy.random.seed(seed)
    dependencies.torch.manual_seed(seed)
    if dependencies.torch.cuda.is_available():
        dependencies.torch.cuda.manual_seed(seed)
        dependencies.torch.cuda.manual_seed_all(seed)
    dependencies.torch.backends.cudnn.deterministic = True
    dependencies.torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _resolve_device(config: GenreTrainingConfig, torch: Any) -> Any:
    name = (
        "cuda:0"
        if config.device == "auto" and torch.cuda.is_available()
        else config.device
    )
    if name == "auto":
        name = "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise GenreTrainingError(
            f"CUDA was requested but is unavailable: {config.device}."
        )
    try:
        return torch.device(name)
    except Exception as error:
        raise GenreTrainingError(f"Invalid training device {name!r}.") from error


def _validation_metrics(
    dependencies: _TrainingDependencies,
    number_of_classes: int,
    device: Any,
) -> dict[str, Any]:
    return {
        "accuracy": dependencies.accuracy(
            num_classes=number_of_classes,
            average="micro",
        ).to(device),
        "roc_auc": dependencies.auroc(
            num_classes=number_of_classes,
            average="macro",
            thresholds=None,
        ).to(device),
        "pr_auc": dependencies.average_precision(
            num_classes=number_of_classes,
            average="macro",
            thresholds=None,
        ).to(device),
    }


def _test_metrics(
    dependencies: _TrainingDependencies,
    number_of_classes: int,
    device: Any,
) -> dict[str, Any]:
    return {
        "accuracy": dependencies.accuracy(
            num_classes=number_of_classes,
            average="micro",
        ).to(device),
        "precision_macro": dependencies.precision(
            num_classes=number_of_classes,
            average="macro",
        ).to(device),
        "recall_macro": dependencies.recall(
            num_classes=number_of_classes,
            average="macro",
        ).to(device),
        "f1_macro": dependencies.f1_score(
            num_classes=number_of_classes,
            average="macro",
        ).to(device),
        "roc_auc_per_class": dependencies.auroc(
            num_classes=number_of_classes,
            average=None,
            thresholds=None,
        ).to(device),
        "pr_auc_per_class": dependencies.average_precision(
            num_classes=number_of_classes,
            average=None,
            thresholds=None,
        ).to(device),
        "confusion_matrix": dependencies.confusion_matrix(
            num_classes=number_of_classes,
        ).to(device),
    }


def _scalar(value: Any, name: str) -> float:
    result = float(value.detach().cpu().item())
    if not math.isfinite(result):
        raise GenreTrainingError(f"Metric {name} is not finite.")
    return result


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, data: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(data, indent=2, sort_keys=True) + "\n",
    )


def _atomic_torch_save(path: Path, data: Any, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(data, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_numpy_archive(path: Path, numpy: Any, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            numpy.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _check_output_targets(config: GenreTrainingConfig) -> None:
    existing = [
        config.output_directory / name
        for name in OUTPUT_FILENAMES
        if (config.output_directory / name).exists()
    ]
    if existing and not config.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise GenreTrainingError(
            f"Training outputs already exist; use --overwrite: {names}"
        )


def _manifest_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_genre_classifier(config: GenreTrainingConfig) -> dict[str, Any]:
    """Train and evaluate the paper-aligned FMA genre-classification MLP."""

    config = _validate_config(config)
    if config.dry_run:
        return plan_genre_training(config)
    _check_output_targets(config)
    records = _read_records(config.manifest_path)
    label_mapping = _label_mapping(records)
    number_of_classes = len(label_mapping)
    dependencies = _load_dependencies()
    _set_seed(config.seed, dependencies)
    device = _resolve_device(config, dependencies.torch)
    LOGGER.info("Using device %s", device)

    prepared = {
        split: _prepare_split(config, records, split, dependencies)
        for split in SPLITS
    }
    _validate_class_coverage(prepared, number_of_classes)
    tensors = {
        split: (
            dependencies.torch.from_numpy(data.features),
            dependencies.torch.from_numpy(data.labels),
        )
        for split, data in prepared.items()
    }
    pin_memory = device.type == "cuda"
    loaders = {
        split: dependencies.data_loader(
            dependencies.tensor_dataset(*tensors[split]),
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        )
        for split in SPLITS
    }

    try:
        from stemfusion.models.mlp import DeepMLPClassifier
    except Exception as error:
        raise GenreTrainingError(f"Could not load the MLP model: {error}") from error

    input_dimension = config.expected_feature_dimension * len(config.sources)
    model = DeepMLPClassifier(
        input_dimension=input_dimension,
        number_of_classes=number_of_classes,
        hidden_dimensions=config.hidden_dimensions,
        dropout=config.dropout,
    ).to(device)
    criterion = dependencies.torch.nn.CrossEntropyLoss()
    optimizer = dependencies.torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )
    validation_metrics = _validation_metrics(
        dependencies,
        number_of_classes,
        device,
    )

    config.output_directory.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        config.output_directory / "resolved_config.json",
        _config_dict(config),
    )
    _atomic_write_text(
        config.output_directory / "data_manifest_checksum.txt",
        f"sha256  {_manifest_checksum(config.manifest_path)}  "
        f"{config.manifest_path}\n",
    )
    _atomic_write_json(
        config.output_directory / "environment.json",
        {
            "numpy": dependencies.numpy.__version__,
            "platform": platform.platform(),
            "python": sys.version,
            "torch": dependencies.torch.__version__,
            "torchmetrics": dependencies.torchmetrics.__version__,
        },
    )

    best_score = -math.inf
    best_epoch = 0
    best_validation: dict[str, float] = {}
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for features, labels in loaders["train"]:
            features = features.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())

        model.eval()
        for metric in validation_metrics.values():
            metric.reset()
        with dependencies.torch.inference_mode():
            for features, labels in loaders["validation"]:
                features = features.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)
                logits = model(features)
                for metric in validation_metrics.values():
                    metric.update(logits, labels)

        validation = {
            name: _scalar(metric.compute(), f"validation_{name}")
            for name, metric in validation_metrics.items()
        }
        combined_score = sum(validation.values())
        average_loss = total_loss / len(loaders["train"])
        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "training_loss": average_loss,
            "validation_accuracy": validation["accuracy"],
            "validation_roc_auc_macro": validation["roc_auc"],
            "validation_pr_auc_macro": validation["pr_auc"],
            "validation_combined_score": combined_score,
        }
        history.append(epoch_record)
        LOGGER.info(
            "Epoch %d/%d | loss %.4f | val accuracy %.4f | val ROC-AUC "
            "%.4f | val PR-AUC %.4f | selection sum %.4f",
            epoch,
            config.epochs,
            average_loss,
            validation["accuracy"],
            validation["roc_auc"],
            validation["pr_auc"],
            combined_score,
        )

        if combined_score > best_score:
            best_score = combined_score
            best_epoch = epoch
            best_validation = validation
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            _atomic_torch_save(
                config.output_directory / "checkpoint.pt",
                {
                    "model_state_dict": best_state,
                    "input_dimension": input_dimension,
                    "number_of_classes": number_of_classes,
                    "label_mapping": label_mapping,
                    "sources": list(config.sources),
                    "best_epoch": best_epoch,
                    "best_validation": best_validation,
                    "selection_score": best_score,
                    "resolved_config": _config_dict(config),
                },
                dependencies.torch,
            )

    if best_state is None:
        raise GenreTrainingError("Training completed without a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    metrics = _test_metrics(dependencies, number_of_classes, device)
    logits_batches: list[Any] = []
    with dependencies.torch.inference_mode():
        for features, labels in loaders["test"]:
            features = features.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)
            logits = model(features)
            logits_batches.append(logits.detach().cpu())
            for metric in metrics.values():
                metric.update(logits, labels)

    accuracy = _scalar(metrics["accuracy"].compute(), "test_accuracy")
    precision = _scalar(
        metrics["precision_macro"].compute(),
        "test_precision_macro",
    )
    recall = _scalar(metrics["recall_macro"].compute(), "test_recall_macro")
    f1 = _scalar(metrics["f1_macro"].compute(), "test_f1_macro")
    roc_per_class = metrics["roc_auc_per_class"].compute().detach().cpu()
    pr_per_class = metrics["pr_auc_per_class"].compute().detach().cpu()
    confusion = metrics["confusion_matrix"].compute().detach().cpu()
    roc_macro = _scalar(roc_per_class.mean(), "test_roc_auc_macro")
    pr_macro = _scalar(pr_per_class.mean(), "test_pr_auc_macro")
    id_to_genre = {label: genre for genre, label in label_mapping.items()}
    per_class = [
        {
            "label": label,
            "genre": id_to_genre[label],
            "roc_auc": float(roc_per_class[label].item()),
            "pr_auc": float(pr_per_class[label].item()),
        }
        for label in range(number_of_classes)
    ]
    test_metrics = {
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "roc_auc_macro": roc_macro,
        "pr_auc_macro": pr_macro,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
    result = {
        "dry_run": False,
        "sources": list(config.sources),
        "source_order": list(config.sources),
        "input_dimension": input_dimension,
        "number_of_classes": number_of_classes,
        "label_mapping": label_mapping,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "validation_selection_score": best_score,
        "selection_criterion": "accuracy + macro ROC-AUC + macro PR-AUC",
        "split_examples": {
            split: len(data.track_ids) for split, data in prepared.items()
        },
        "dropped_missing_features": {
            split: data.dropped for split, data in prepared.items()
        },
        "test": test_metrics,
    }
    _atomic_write_json(config.output_directory / "metrics.json", result)
    _atomic_write_json(
        config.output_directory / "training_history.json",
        history,
    )
    logits = dependencies.torch.cat(logits_batches).numpy()
    predictions = logits.argmax(axis=1)
    _atomic_numpy_archive(
        config.output_directory / "predictions.npz",
        dependencies.numpy,
        track_ids=dependencies.numpy.asarray(prepared["test"].track_ids),
        targets=prepared["test"].labels,
        predictions=predictions,
        logits=logits,
    )
    return result


def _parse_assignment(value: str, option: str) -> tuple[str, str]:
    name, separator, assigned = value.partition("=")
    if not separator or not name.strip() or not assigned.strip():
        raise GenreTrainingError(
            f"{option} must use SOURCE=PATH syntax: {value!r}."
        )
    return name.strip(), assigned.strip()


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Train the paper-aligned three-layer MLP on cached FMA MusicFM "
        "features. Repeat --source to run a mid-fusion combination."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help=(
            "Feature source; repeat for fusion. Canonical order is mixture, "
            "guitar, drums, piano, bass, vocals, other."
        ),
    )
    parser.add_argument(
        "--source-feature-dir",
        action="append",
        default=[],
        metavar="SOURCE=PATH",
        help=(
            "Override one source's cache directory. This also enables the "
            "legacy fma_musicfm_SPLIT.pkl filenames in that directory."
        ),
    )
    parser.add_argument("--expected-feature-dimension", type=int, default=1024)
    parser.add_argument(
        "--hidden-dimensions",
        type=int,
        nargs=2,
        default=(1024, 512),
        metavar=("FIRST", "SECOND"),
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-missing-features", action="store_true")
    parser.add_argument("--drop-silent-features", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(command_handler=run_from_args)


def config_from_args(args: argparse.Namespace) -> GenreTrainingConfig:
    sources = canonicalize_sources(args.source)
    overrides: list[SourceFeatureDirectory] = []
    for value in args.source_feature_dir:
        source, directory = _parse_assignment(value, "--source-feature-dir")
        overrides.append(
            SourceFeatureDirectory(
                source=normalize_source(source),
                directory=Path(directory).expanduser().resolve(),
            )
        )
    return GenreTrainingConfig(
        manifest_path=args.manifest.expanduser().resolve(),
        feature_directory=args.feature_dir.expanduser().resolve(),
        output_directory=args.output_dir.expanduser().resolve(),
        sources=sources,
        source_feature_directories=tuple(overrides),
        expected_feature_dimension=args.expected_feature_dimension,
        hidden_dimensions=tuple(args.hidden_dimensions),
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        allow_missing_features=args.allow_missing_features,
        drop_silent_features=args.drop_silent_features,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


def run_from_args(args: argparse.Namespace) -> int:
    try:
        result = train_genre_classifier(config_from_args(args))
    except GenreTrainingError as error:
        print(f"Genre training error: {error}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0

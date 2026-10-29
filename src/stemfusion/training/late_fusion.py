"""Shared unweighted late-fusion runner retained for FMA-Medium migration."""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from stemfusion.data.fma import FMAManifestRecord
from stemfusion.training.genre import (
    OUTPUT_FILENAMES,
    SPLITS,
    GenreTrainingConfig,
    GenreTrainingError,
    SourceFeatureDirectory,
    _atomic_numpy_archive,
    _atomic_torch_save,
    _atomic_write_json,
    _atomic_write_text,
    _feature_vector,
    _label_mapping,
    _load_dependencies,
    _load_feature_cache,
    _manifest_checksum,
    _parse_assignment,
    _read_records,
    _resolve_device,
    _scalar,
    _set_seed,
    _test_metrics,
    _validate_class_coverage,
    _validate_config,
    _validation_metrics,
    canonicalize_sources,
    normalize_source,
    resolve_feature_cache_path,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LateFusionTrainingConfig:
    """Settings for the retained unweighted source-MLP runner."""

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
    overwrite: bool = False
    dry_run: bool = False
    config_provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _PreparedSplit:
    features: Any
    labels: Any
    track_ids: tuple[str, ...]
    dropped: int


def _base_config(config: LateFusionTrainingConfig) -> GenreTrainingConfig:
    return GenreTrainingConfig(
        manifest_path=config.manifest_path,
        feature_directory=config.feature_directory,
        output_directory=config.output_directory,
        sources=config.sources,
        source_feature_directories=config.source_feature_directories,
        expected_feature_dimension=config.expected_feature_dimension,
        hidden_dimensions=config.hidden_dimensions,
        dropout=config.dropout,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        seed=config.seed,
        device=config.device,
        num_workers=config.num_workers,
        allow_missing_features=config.allow_missing_features,
        overwrite=config.overwrite,
        dry_run=config.dry_run,
        config_provenance=config.config_provenance,
    )


def _validate_config_late(
    config: LateFusionTrainingConfig,
) -> LateFusionTrainingConfig:
    sources = canonicalize_sources(config.sources)
    if sources != config.sources:
        config = replace(config, sources=sources)
    _validate_config(_base_config(config))
    return config


def _prepare_split(
    config: LateFusionTrainingConfig,
    records: Sequence[FMAManifestRecord],
    split: str,
    dependencies: Any,
) -> _PreparedSplit:
    base_config = _base_config(config)
    split_records = [record for record in records if record.split == split]
    feature_caches = {
        source: _load_feature_cache(
            resolve_feature_cache_path(base_config, source, split)
        )
        for source in config.sources
    }
    feature_rows: list[Any] = []
    labels: list[int] = []
    track_ids: list[str] = []
    missing_counts: Counter[str] = Counter()
    missing_examples: list[str] = []

    for record in split_records:
        missing = [
            source
            for source in config.sources
            if record.track_id not in feature_caches[source]
        ]
        if missing:
            missing_counts.update(missing)
            if len(missing_examples) < 10:
                missing_examples.append(
                    f"{record.track_id} ({', '.join(missing)})"
                )
            continue
        vectors = [
            _feature_vector(
                feature_caches[source][record.track_id],
                source,
                record.track_id,
                config.expected_feature_dimension,
                dependencies.numpy,
            )
            for source in config.sources
        ]
        feature_rows.append(dependencies.numpy.stack(vectors))
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
            "only to reproduce legacy intersections."
        )
    if not feature_rows:
        raise GenreTrainingError(
            f"No aligned examples remain in the {split} split."
        )
    features = dependencies.numpy.stack(feature_rows).astype(
        dependencies.numpy.float32,
        copy=False,
    )
    expected_features = (
        len(config.sources),
        config.expected_feature_dimension,
    )
    if features.shape[1:] != expected_features:
        raise GenreTrainingError(
            f"Expected per-track feature shape {expected_features}, received "
            f"{tuple(features.shape[1:])}."
        )
    return _PreparedSplit(
        features=features,
        labels=dependencies.numpy.asarray(labels, dtype=dependencies.numpy.int64),
        track_ids=tuple(track_ids),
        dropped=len(split_records) - len(track_ids),
    )


def _config_dict(config: LateFusionTrainingConfig) -> dict[str, Any]:
    return {
        "config_provenance": config.config_provenance,
        "manifest_path": str(config.manifest_path),
        "feature_directory": str(config.feature_directory),
        "output_directory": str(config.output_directory),
        "sources": list(config.sources),
        "fusion": "unweighted_late",
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
        "overwrite": config.overwrite,
    }


def plan_late_fusion_training(config: LateFusionTrainingConfig) -> dict[str, Any]:
    """Validate paths and describe a late-fusion run without loading caches."""

    config = _validate_config_late(config)
    base_config = _base_config(config)
    records = _read_records(config.manifest_path)
    cache_paths = {
        source: {
            split: str(resolve_feature_cache_path(base_config, source, split))
            for split in SPLITS
        }
        for source in config.sources
    }
    counts = Counter(record.split for record in records)
    return {
        "dry_run": True,
        "fusion": "unweighted_late",
        "sources": list(config.sources),
        "source_order": list(config.sources),
        "per_source_feature_dimension": config.expected_feature_dimension,
        "number_of_classes": len(_label_mapping(records)),
        "manifest_split_counts": {split: counts[split] for split in SPLITS},
        "feature_caches": cache_paths,
        "output_directory": str(config.output_directory),
        "training": _config_dict(config),
    }


def _check_output_targets(config: LateFusionTrainingConfig) -> None:
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


def _build_model(config: LateFusionTrainingConfig, number_of_classes: int) -> Any:
    try:
        from stemfusion.models.late_fusion import UnweightedLateFusionMLP
    except Exception as error:
        raise GenreTrainingError(
            f"Could not load the late-fusion model: {error}"
        ) from error
    return UnweightedLateFusionMLP(
        number_of_sources=len(config.sources),
        feature_dimension=config.expected_feature_dimension,
        number_of_classes=number_of_classes,
        hidden_dimensions=config.hidden_dimensions,
        dropout=config.dropout,
    )


def _run_model(model: Any, batch: Sequence[Any]) -> tuple[Any, Any]:
    features, labels = batch
    return model(features), labels


def train_late_fusion_classifier(
    config: LateFusionTrainingConfig,
) -> dict[str, Any]:
    """Train the retained model jointly through summed source logits.

    This historical runner does not yet implement the revised protocol's
    independently trained and individually selected branch checkpoints.
    """

    config = _validate_config_late(config)
    if config.dry_run:
        return plan_late_fusion_training(config)
    _check_output_targets(config)
    records = _read_records(config.manifest_path)
    label_mapping = _label_mapping(records)
    number_of_classes = len(label_mapping)
    dependencies = _load_dependencies()
    _set_seed(config.seed, dependencies)
    device = _resolve_device(_base_config(config), dependencies.torch)
    pin_memory = device.type == "cuda"
    prepared = {
        split: _prepare_split(config, records, split, dependencies)
        for split in SPLITS
    }
    coverage = {
        split: replace(
            data,
            features=data.features.reshape(len(data.features), -1),
        )
        for split, data in prepared.items()
    }
    _validate_class_coverage(coverage, number_of_classes)
    datasets = {}
    for split, data in prepared.items():
        tensors = [dependencies.torch.from_numpy(data.features)]
        tensors.append(dependencies.torch.from_numpy(data.labels))
        datasets[split] = dependencies.tensor_dataset(*tensors)
    loaders = {
        split: dependencies.data_loader(
            datasets[split],
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        )
        for split in SPLITS
    }
    model = _build_model(config, number_of_classes).to(device)
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
        for batch in loaders["train"]:
            batch = tuple(
                value.to(device, non_blocking=pin_memory) for value in batch
            )
            optimizer.zero_grad()
            logits, labels = _run_model(model, batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
        model.eval()
        for metric in validation_metrics.values():
            metric.reset()
        with dependencies.torch.inference_mode():
            for batch in loaders["validation"]:
                batch = tuple(
                    value.to(device, non_blocking=pin_memory) for value in batch
                )
                logits, labels = _run_model(model, batch)
                for metric in validation_metrics.values():
                    metric.update(logits, labels)
        validation = {
            name: _scalar(metric.compute(), f"validation_{name}")
            for name, metric in validation_metrics.items()
        }
        selection_score = sum(validation.values())
        average_loss = total_loss / len(loaders["train"])
        history.append(
            {
                "epoch": epoch,
                "training_loss": average_loss,
                "validation_accuracy": validation["accuracy"],
                "validation_roc_auc_macro": validation["roc_auc"],
                "validation_pr_auc_macro": validation["pr_auc"],
                "validation_combined_score": selection_score,
            }
        )
        LOGGER.info(
            "Epoch %d/%d | loss %.4f | val accuracy %.4f | val ROC-AUC "
            "%.4f | val PR-AUC %.4f | selection sum %.4f",
            epoch,
            config.epochs,
            average_loss,
            validation["accuracy"],
            validation["roc_auc"],
            validation["pr_auc"],
            selection_score,
        )
        if selection_score > best_score:
            best_score = selection_score
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
                    "fusion": "unweighted_late",
                    "number_of_sources": len(config.sources),
                    "feature_dimension": config.expected_feature_dimension,
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
        for batch in loaders["test"]:
            batch = tuple(
                value.to(device, non_blocking=pin_memory) for value in batch
            )
            logits, labels = _run_model(model, batch)
            logits_batches.append(logits.detach().cpu())
            for metric in metrics.values():
                metric.update(logits, labels)
    accuracy = _scalar(metrics["accuracy"].compute(), "test_accuracy")
    precision = _scalar(metrics["precision_macro"].compute(), "test_precision_macro")
    recall = _scalar(metrics["recall_macro"].compute(), "test_recall_macro")
    f1 = _scalar(metrics["f1_macro"].compute(), "test_f1_macro")
    roc_per_class = metrics["roc_auc_per_class"].compute().detach().cpu()
    pr_per_class = metrics["pr_auc_per_class"].compute().detach().cpu()
    confusion = metrics["confusion_matrix"].compute().detach().cpu()
    roc_macro = _scalar(roc_per_class.mean(), "test_roc_auc_macro")
    pr_macro = _scalar(pr_per_class.mean(), "test_pr_auc_macro")
    id_to_genre = {label: genre for genre, label in label_mapping.items()}
    test_metrics = {
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "roc_auc_macro": roc_macro,
        "pr_auc_macro": pr_macro,
        "per_class": [
            {
                "label": label,
                "genre": id_to_genre[label],
                "roc_auc": float(roc_per_class[label].item()),
                "pr_auc": float(pr_per_class[label].item()),
            }
            for label in range(number_of_classes)
        ],
        "confusion_matrix": confusion.tolist(),
    }
    result = {
        "dry_run": False,
        "fusion": "unweighted_late",
        "sources": list(config.sources),
        "source_order": list(config.sources),
        "per_source_feature_dimension": config.expected_feature_dimension,
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
    _atomic_numpy_archive(
        config.output_directory / "predictions.npz",
        dependencies.numpy,
        track_ids=dependencies.numpy.asarray(prepared["test"].track_ids),
        targets=prepared["test"].labels,
        predictions=logits.argmax(axis=1),
        logits=logits,
    )
    return result


def configure_parser(
    parser: argparse.ArgumentParser,
    *,
    command_handler: Any,
) -> None:
    """Configure an unweighted late-fusion command."""

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
        help="Override one MusicFM source cache directory.",
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(command_handler=command_handler)


def config_from_args(args: argparse.Namespace) -> LateFusionTrainingConfig:
    source_directories = tuple(
        SourceFeatureDirectory(
            source=normalize_source(_parse_assignment(value, "--source-feature-dir")[0]),
            directory=Path(_parse_assignment(value, "--source-feature-dir")[1])
            .expanduser()
            .resolve(),
        )
        for value in args.source_feature_dir
    )
    return LateFusionTrainingConfig(
        manifest_path=args.manifest.expanduser().resolve(),
        feature_directory=args.feature_dir.expanduser().resolve(),
        output_directory=args.output_dir.expanduser().resolve(),
        sources=canonicalize_sources(args.source),
        source_feature_directories=source_directories,
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
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        config_provenance=getattr(args, "yaml_provenance", {}),
    )


def run_from_args(args: argparse.Namespace) -> int:
    try:
        result = train_late_fusion_classifier(
            config_from_args(args)
        )
    except GenreTrainingError as error:
        LOGGER.error("%s", error)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0

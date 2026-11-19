"""Plain CNN emotion experiments through the unchanged binary Music2Emo fitter."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import logging
import math
from pathlib import Path

import yaml

from stemfusion.configuration import JAMENDO_MID_CONDITIONS, _UniqueKeyLoader, load_configuration
from stemfusion.data.emotion_cnn_features import (
    TARGETS, JamendoCNNDataset, load_emotion_cnn_features, manifest_reader,
)
from stemfusion.models.cnn_emotion import CNNEmotionClassifier, parameter_count
from stemfusion.training import music2emo
from stemfusion.training.emotion import BinaryEmotionTask
from stemfusion.training.reporting import file_checksum, fingerprint, summarize_results, write_json

LOGGER = logging.getLogger(__name__)
SEEDS = (32, 64, 128, 256, 512)
DATASETS = {"pmemo": "PMEmo", "deam": "DEAM", "emomusic": "EmoMusic"}


@dataclass(frozen=True)
class EmotionCNNConfig:
    manifest: Path
    feature_dir: Path
    output_dir: Path
    summary_dir: Path
    conditions: dict
    label_policy: dict
    dataset: str = "pmemo"
    targets: tuple[str, ...] = TARGETS
    seeds: tuple[int, ...] = SEEDS
    epochs: int = 200
    batch_size: int = 8
    learning_rate: float = 1e-4
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    device: str = "cuda"
    num_workers: int = 4
    resume: bool = True
    overwrite: bool = False
    dry_run: bool = False
    validate_only: bool = False
    selected_run: str | None = None
    config_provenance: dict | None = None


def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    names = ("experiments/_cnn_emotion.py", "data/emotion_cnn_features.py", "data/jamendo_cnn_features.py",
             "models/cnn_emotion.py", "training/music2emo.py", "training/emotion.py",
             "training/trainer.py", "training/reporting.py")
    return fingerprint({name: file_checksum(root / name) for name in names})


def config_from_args(args, *, dataset="pmemo"):
    if dataset not in DATASETS:
        raise ValueError("Unsupported CNN emotion dataset")
    path = args.config.resolve()
    raw = yaml.load(path.read_text(), Loader=_UniqueKeyLoader)
    if not isinstance(raw, dict) or set(raw) != {"protocol", "dataset", "features", "model", "paths"} or \
       raw["dataset"] != dataset or raw["features"] != {
           "name": "msd_musicnn", "dimension_per_source": 200, "pooling": "track_window_mean", "harmony": False} or \
       raw["model"] != {"hidden_dimensions": [512, 256], "activation": "relu", "dropout": 0., "output_labels": 1}:
        raise ValueError("Expected 200-d CNN features and a plain 512/256/1 ReLU head without dropout")
    role = "emotion_mid_fusion" if dataset == "pmemo" else f"{dataset}_mid_fusion"
    defaults, provenance = load_configuration((path.parent / raw["protocol"]).resolve(), role)
    task = provenance["yaml_settings_before_cli_overrides"]["task"]
    scales = {"zero_one": "original_0_1", "one_nine": "original_1_9"}
    label_policy = {"annotation_scale": scales[task["score_scale"]], "threshold": task["label_threshold"],
                    "positive_comparison": ">=", "negative_class": "low", "positive_class": "high"}
    if set(raw["paths"]) != {"manifest", "feature_dir", "output_dir", "summary_dir"}:
        raise ValueError("Missing or unsupported CNN emotion paths")
    paths = {key: (path.parent / value).resolve() for key, value in raw["paths"].items()}
    overrides = {}
    for key in paths:
        if getattr(args, key) is not None:
            paths[key] = getattr(args, key).expanduser().resolve()
            overrides[key] = str(paths[key])
    if args.output_dir is not None and args.summary_dir is None:
        paths["summary_dir"] = paths["output_dir"]
    settings = {key: defaults[key] for key in (
        "targets", "seeds", "epochs", "batch_size", "learning_rate", "early_stopping_patience",
        "early_stopping_min_delta", "device", "num_workers", "resume", "overwrite", "dry_run", "validate_only",
    )}
    for key in settings:
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
            overrides[key] = settings[key]
    settings["targets"], settings["seeds"] = tuple(settings["targets"]), tuple(settings["seeds"])
    if settings["overwrite"]:
        settings["resume"] = False
    provenance.update(cnn_config={"path": str(path), "sha256": file_checksum(path)}, cli_overrides=overrides,
                      representation_override=raw["features"], classifier_override=raw["model"],
                      resolved_cnn_paths={key: str(value) for key, value in paths.items()})
    return EmotionCNNConfig(**paths, **settings, conditions=defaults["runs"], label_policy=label_policy,
                            dataset=dataset, selected_run=args.run, config_provenance=provenance)


def plan_experiment(config):
    if config.dataset not in DATASETS or not config.targets or len(set(config.targets)) != len(config.targets) or \
       set(config.targets) - set(TARGETS):
        raise ValueError("Choose a supported dataset and distinct valence/arousal targets")
    if not config.seeds or len(set(config.seeds)) != len(config.seeds) or any(
        type(seed) is not int or not 0 <= seed < 2**32 for seed in config.seeds
    ):
        raise ValueError("Choose distinct valid integer outer seeds")
    if config.epochs < 1 or config.batch_size < 1 or config.num_workers < 0 or \
       not math.isfinite(config.learning_rate) or config.learning_rate <= 0 or \
       type(config.early_stopping_patience) is not int or config.early_stopping_patience < 0 or \
       not math.isfinite(config.early_stopping_min_delta) or config.early_stopping_min_delta < 0:
        raise ValueError("Invalid training or early-stopping settings")
    if config.conditions != JAMENDO_MID_CONDITIONS:
        raise ValueError("Use the shared fixed seven-condition mid-fusion matrix")
    if config.selected_run is not None and config.selected_run not in config.conditions:
        raise ValueError(f"Unknown condition: {config.selected_run}")
    selected = ({config.selected_run: config.conditions[config.selected_run]} if config.selected_run else config.conditions)
    conditions = {}
    for name, condition in selected.items():
        count = condition.get("matched_sources", len(condition["sources"]))
        conditions[name] = {**condition, "input_dimension": count * 200,
                            "trainable_parameters": parameter_count(count)}
    return {"dataset": config.dataset, "targets": list(config.targets), "features": "MSD_musicnn",
            "harmony_features": False, "seeds": list(config.seeds), "conditions": conditions,
            "planned_fits": len(config.targets) * len(config.seeds) * len(conditions),
            "hidden_dimensions": [512, 256], "output_labels": 1, "dropout": 0.,
            "label_policy": config.label_policy, "epochs": config.epochs, "batch_size": config.batch_size,
            "learning_rate": config.learning_rate, "early_stopping_patience": config.early_stopping_patience,
            "early_stopping_min_delta": config.early_stopping_min_delta,
            "checkpoint": "minimum_validation_weighted_bce", "threshold": "validation_macro_f1_global_grid",
            "manifest": str(config.manifest), "feature_dir": str(config.feature_dir),
            "output_dir": str(config.output_dir), "summary_dir": str(config.summary_dir)}


def summary_path(config, target):
    name = f"summary_mid_fusion_{target}"
    if config.selected_run:
        name += f"_run_{config.selected_run}"
    if config.seeds != SEEDS:
        name += "_seeds_" + "_".join(map(str, config.seeds))
    return config.summary_dir / f"{name}.json"


def format_summary(summary):
    def render(value, scale=1.):
        text = f"{value['mean'] * scale:.4f}"
        return text + (f" +/- {value['sample_sd'] * scale:.4f}" if value["sample_sd"] is not None else " (SD unavailable)")
    def cells(metrics):
        return [render(metrics[key], 100 if key == "accuracy" else 1)
                for key in ("accuracy", "f1_macro", "roc_auc_macro", "pr_auc_macro")]
    policy = summary["label_policy"]
    lines = [f"{DATASETS[summary['dataset']]}: binary {summary['target']} | frozen MSD_musicnn mid-fusion",
             "Seeds: " + ", ".join(map(str, summary["seeds"])) + "; mean +/- sample SD (ddof=1).",
             f"Ground truth: {policy['annotation_scale']} score >= {policy['threshold']} is high; lower scores are low.",
             "Features: fixed track-window means, 200 dimensions per source; plain 512/256/1 ReLU head.",
             "No chord/key features, harmonic Transformer, or head dropout.",
             "Checkpoint: minimum validation weighted BCE; threshold: validation macro-F1 grid, ties to lowest.",
             "Accuracy is track-level; macro metrics average low and high classes. PR-AUC means average precision.", "",
             "Condition | Accuracy (%) | Macro F1 | Macro ROC-AUC | Macro PR-AUC", "--- | --- | --- | --- | ---"]
    for name, metrics in summary["conditions"].items():
        lines.append(" | ".join([name, *cells(metrics)]))
    if summary["paired_differences"]:
        lines += ["", "Paired differences: fusion minus its repeated-mixture control.",
                  "Comparison | Accuracy (percentage points) | Macro F1 | Macro ROC-AUC | Macro PR-AUC",
                  "--- | --- | --- | --- | ---"]
        for name, comparison in summary["paired_differences"].items():
            lines.append(" | ".join([f"{name} - {comparison['control']}", *cells(comparison["metrics"])]))
    return "\n".join(lines) + "\n"


def run_experiment(config, *, features=None):
    plan = plan_experiment(config)
    if config.dry_run:
        return {"dry_run": True, **plan}
    if features is None:
        features = load_emotion_cnn_features(config.manifest, config.feature_dir,
                                            dataset=config.dataset, label_policy=config.label_policy)
    target_data = {target: features.target(target) for target in config.targets}
    if config.validate_only:
        return {"validate_only": True, "plan": plan, "targets": {target: data.audit for target, data in target_data.items()}}
    implementation = implementation_identity()
    summaries = {}
    for target, data in target_data.items():
        task = BinaryEmotionTask(target)
        results = {name: {} for name in plan["conditions"]}
        for seed in config.seeds:
            for name, condition in plan["conditions"].items():
                sources = tuple(condition["sources"])
                copies = condition.get("matched_sources", 1)
                count = condition.get("matched_sources", len(sources))
                fit_config = replace(config, targets=(target,), output_dir=config.output_dir / target / name)
                result = music2emo.fit_model(
                    fit_config, data, seed,
                    model_factory=lambda count=count: CNNEmotionClassifier(count),
                    dataset_factory=lambda records, training, sources=sources, copies=copies:
                        JamendoCNNDataset(records, sources=sources, copies=copies),
                    implementation=implementation, task=task,
                    protocol_updates={"condition": name, "source": list(sources), "inputs": count,
                                      "kind": condition["kind"], "feature_backend": "msd_musicnn",
                                      "feature_dimension_per_source": 200, "hidden_dimensions": [512, 256],
                                      "head_dropout": 0., "harmony_features": False,
                                      "sampling": "fixed_track_window_mean", "repeated_mixture_copies": copies,
                                      "label_policy": config.label_policy})
                if result["trainable_parameters"] != condition["trainable_parameters"]:
                    raise ValueError("Actual CNN parameters do not match the experiment plan")
                results[name][seed] = {**result, "condition": name, "target": target}
        comparisons = {name: f"repeated_mixture_{len(condition['sources'])}" for name, condition in plan["conditions"].items()
                       if condition["kind"] == "fusion"}
        summary = summarize_results(results, config.seeds, comparisons)
        summary.update(dataset=config.dataset, target=target, method="mid_fusion", features="msd_musicnn",
                       runs=results, plan=plan, data_audit=data.audit, label_policy=config.label_policy,
                       metric_conventions=task.protocol(), implementation_sha256=implementation,
                       config_provenance=config.config_provenance,
                       parameter_counts={name: value["trainable_parameters"] for name, value in plan["conditions"].items()})
        path = summary_path(config, target)
        write_json(path, summary)
        temporary = path.with_suffix(".txt.tmp")
        try:
            temporary.write_text(format_summary(summary), encoding="utf-8")
            temporary.replace(path.with_suffix(".txt"))
        finally:
            temporary.unlink(missing_ok=True)
        summaries[target] = summary
    return {"targets": summaries, "plan": plan}


def main(argv=None, *, dataset="pmemo"):
    if dataset not in DATASETS:
        raise ValueError("Unsupported CNN emotion dataset")
    default = Path(__file__).resolve().parents[3] / f"configs/experiments/{dataset}/mid_fusion_cnn.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default)
    for name in ("manifest", "feature-dir", "output-dir", "summary-dir"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--target", choices=TARGETS, action="append", dest="targets")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--run")
    for name in ("epochs", "batch-size", "num-workers", "early-stopping-patience"):
        parser.add_argument(f"--{name}", type=int)
    for name in ("learning-rate", "early-stopping-min-delta"):
        parser.add_argument(f"--{name}", type=float)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--validate-only", action="store_true", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", default=None)
    mode.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        config = config_from_args(args, dataset=dataset)
        result = run_experiment(config)
        if config.dry_run or config.validate_only:
            print(json.dumps(result, indent=2, default=str))
        else:
            for target, summary in result["targets"].items():
                print(format_summary(summary), end="")
                print(f"\nReadable report: {summary_path(config, target).with_suffix('.txt')}\nDetailed results: {summary_path(config, target)}")
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        LOGGER.error("%s", error)
        return 2
    return 0

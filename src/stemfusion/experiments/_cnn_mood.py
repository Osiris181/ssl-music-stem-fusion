"""Plain CNN mood/theme experiments using the existing Music2Emo fitter."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import logging
import math
from pathlib import Path

import yaml

from stemfusion.configuration import JAMENDO_MID_CONDITIONS, _UniqueKeyLoader, load_configuration
from stemfusion.data.jamendo_cnn_features import JamendoCNNDataset, load_jamendo_cnn_features
from stemfusion.models.cnn_mood import CNNMoodClassifier
from stemfusion.training import music2emo
from stemfusion.training.mid_fusion import model_parameter_count
from stemfusion.training.reporting import file_checksum, fingerprint, summarize_results, write_json

LOGGER = logging.getLogger(__name__)
SEEDS = (32, 64, 128, 256, 512)
DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs/experiments/mtg_jamendo/mid_fusion_cnn.yaml"


@dataclass(frozen=True)
class JamendoCNNConfig:
    manifest: Path
    feature_dir: Path
    tag_list: Path
    prevalence: Path
    output_dir: Path
    summary_dir: Path
    conditions: dict
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
    names = ("experiments/_cnn_mood.py", "data/jamendo_cnn_features.py", "models/cnn_mood.py",
             "models/mlp.py", "training/music2emo.py", "training/trainer.py", "training/reporting.py")
    return fingerprint({name: file_checksum(root / name) for name in names})


def config_from_args(args):
    path = args.config.resolve()
    raw = yaml.load(path.read_text(), Loader=_UniqueKeyLoader)
    if not isinstance(raw, dict) or set(raw) != {"protocol", "dataset", "features", "model", "paths"} or \
       raw["dataset"] != "mtg_jamendo" or raw["features"] != {
           "name": "msd_musicnn", "dimension_per_source": 200, "pooling": "track_window_mean", "harmony": False} or \
       raw["model"] != {"hidden_dimensions": [512, 256], "activation": "relu", "dropout": 0., "output_labels": 56}:
        raise ValueError("Expected Jamendo 200-d CNN features and a plain 512/256/56 ReLU head without dropout")
    defaults, provenance = load_configuration((path.parent / raw["protocol"]).resolve(), "jamendo_mid_fusion")
    if set(raw["paths"]) != {"manifest", "feature_dir", "tag_list", "prevalence", "output_dir", "summary_dir"}:
        raise ValueError("Missing or unsupported Jamendo CNN training paths")
    paths = {key: (path.parent / value).resolve() for key, value in raw["paths"].items()}
    overrides = {}
    for key in paths:
        if getattr(args, key) is not None:
            paths[key] = getattr(args, key).resolve()
            overrides[key] = str(paths[key])
    if args.output_dir is not None and args.summary_dir is None:
        paths["summary_dir"] = paths["output_dir"]
    settings = {key: defaults[key] for key in
                ("seeds", "epochs", "batch_size", "learning_rate", "early_stopping_patience",
                 "early_stopping_min_delta", "device", "num_workers", "resume", "overwrite", "dry_run", "validate_only")}
    for key in settings:
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
            overrides[key] = settings[key]
    settings["seeds"] = tuple(settings["seeds"])
    if settings["overwrite"]:
        settings["resume"] = False
    provenance.update(cnn_config={"path": str(path), "sha256": file_checksum(path)}, cli_overrides=overrides,
                      representation_override=raw["features"], classifier_override=raw["model"],
                      resolved_cnn_paths={key: str(value) for key, value in paths.items()})
    return JamendoCNNConfig(**paths, **settings, conditions=defaults["runs"],
                            selected_run=args.run, config_provenance=provenance)


def plan_experiment(config):
    if not config.seeds or len(set(config.seeds)) != len(config.seeds) or any(
        type(seed) is not int or not 0 <= seed < 2**32 for seed in config.seeds
    ):
        raise ValueError("Choose distinct valid integer outer seeds")
    if config.epochs < 1 or config.batch_size < 1 or config.num_workers < 0 or \
       not math.isfinite(config.learning_rate) or config.learning_rate <= 0 or \
       type(config.early_stopping_patience) is not int or config.early_stopping_patience < 0 or \
       not math.isfinite(config.early_stopping_min_delta) or config.early_stopping_min_delta < 0:
        raise ValueError("Invalid training or early-stopping settings")
    if not config.conditions or any(JAMENDO_MID_CONDITIONS.get(name) != value for name, value in config.conditions.items()):
        raise ValueError("Use the existing seven Jamendo mid-fusion conditions")
    if config.selected_run is not None and config.selected_run not in config.conditions:
        raise ValueError(f"Unknown condition: {config.selected_run}")
    selected = ({config.selected_run: config.conditions[config.selected_run]} if config.selected_run else config.conditions)
    conditions = {}
    for name, condition in selected.items():
        count = condition.get("matched_sources", len(condition["sources"]))
        conditions[name] = {**condition, "input_dimension": count * 200,
                            "trainable_parameters": model_parameter_count(count * 200, 56, (512, 256))}
    return {"dataset": "mtg_jamendo", "features": "MSD_musicnn", "harmony_features": False,
            "seeds": list(config.seeds), "conditions": conditions, "hidden_dimensions": [512, 256],
            "output_labels": 56, "dropout": 0., "epochs": config.epochs, "batch_size": config.batch_size,
            "learning_rate": config.learning_rate, "early_stopping_patience": config.early_stopping_patience,
            "early_stopping_min_delta": config.early_stopping_min_delta,
            "checkpoint": "minimum_validation_weighted_bce", "threshold": "validation_macro_f1_global_grid",
            "manifest": str(config.manifest), "feature_dir": str(config.feature_dir),
            "output_dir": str(config.output_dir), "summary_dir": str(config.summary_dir)}


def summary_path(config):
    name = "summary_mid_fusion"
    if config.selected_run:
        name += f"_run_{config.selected_run}"
    if config.seeds != SEEDS:
        name += "_seeds_" + "_".join(map(str, config.seeds))
    return config.summary_dir / f"{name}.json"


def format_summary(summary):
    def render(metric, scale=1.):
        text = f"{metric['mean'] * scale:.4f}"
        return text + (f" +/- {metric['sample_sd'] * scale:.4f}" if metric["sample_sd"] is not None else " (SD unavailable)")
    def cells(metrics):
        return [render(metrics[key], 100 if key == "accuracy" else 1)
                for key in ("accuracy", "f1_macro", "roc_auc_macro", "pr_auc_macro")]
    lines = ["MTG-Jamendo: frozen MSD_musicnn mid-fusion; plain 512/256 ReLU head",
             "Seeds: " + ", ".join(map(str, summary["seeds"])) + "; mean +/- sample SD (ddof=1).",
             "Features: track-window mean, 200 dimensions per source; no chord/key features or Transformer.",
             "Checkpoint: minimum validation weighted BCE; patience 10/min_delta 1e-4 by default.",
             "Accuracy is label-wise. One global threshold maximizes validation macro F1.",
             "PR-AUC = macro average precision; all metrics except accuracy are on the 0-1 scale.", "",
             "Condition | Accuracy (%) | Macro F1 | Macro ROC-AUC | Macro PR-AUC",
             "--- | --- | --- | --- | ---"]
    for name, metrics in summary["conditions"].items():
        lines.append(" | ".join([name, *cells(metrics)]))
    if summary["paired_differences"]:
        lines += ["", "Paired differences: fusion minus its repeated-mixture control.",
                  "Comparison | Accuracy (percentage points) | Macro F1 | Macro ROC-AUC | Macro PR-AUC",
                  "--- | --- | --- | --- | ---"]
        for name, comparison in summary["paired_differences"].items():
            lines.append(" | ".join([f"{name} - {comparison['control']}", *cells(comparison["metrics"])]))
    return "\n".join(lines) + "\n"


def run_experiment(config):
    plan = plan_experiment(config)
    if config.dry_run:
        return plan
    data = load_jamendo_cnn_features(config.manifest, config.feature_dir, config.tag_list, config.prevalence)
    if config.validate_only:
        return {"validate_only": True, "audit": data.audit, "plan": plan}
    implementation = implementation_identity()
    results = {name: {} for name in plan["conditions"]}
    for seed in config.seeds:
        for name, condition in plan["conditions"].items():
            sources = tuple(condition["sources"])
            copies = condition.get("matched_sources", 1)
            count = condition.get("matched_sources", len(sources))
            fit_config = replace(config, output_dir=config.output_dir / name)
            result = music2emo.fit_model(
                fit_config, data, seed,
                model_factory=lambda count=count: CNNMoodClassifier(count),
                dataset_factory=lambda records, training, sources=sources, copies=copies:
                    JamendoCNNDataset(records, sources=sources, copies=copies),
                implementation=implementation,
                protocol_updates={"task": "jamendo_cnn_mid_fusion", "condition": name, "source": list(sources),
                                  "inputs": count, "kind": condition["kind"], "feature_dimension_per_source": 200,
                                  "hidden_dimensions": [512, 256], "head_dropout": 0., "harmony_features": False,
                                  "sampling": "fixed_track_window_mean", "repeated_mixture_copies": copies,
                                  "threshold": "validation_macro_f1_global_0_to_1_step_0.01_ties_lowest",
                                  "accuracy": "labelwise"})
            if result["trainable_parameters"] != condition["trainable_parameters"]:
                raise ValueError("Actual CNN parameter count does not match the experiment plan")
            results[name][seed] = result
    comparisons = {name: f"repeated_mixture_{len(condition['sources'])}" for name, condition in plan["conditions"].items()
                   if condition["kind"] == "fusion"}
    summary = summarize_results(results, config.seeds, comparisons)
    summary.update(dataset="mtg_jamendo", method="mid_fusion", features="msd_musicnn", runs=results,
                   plan=plan, data_audit=data.audit, accuracy_definition="labelwise",
                   threshold_policy="validation_macro_f1_global_grid", implementation_sha256=implementation,
                   config_provenance=config.config_provenance,
                   parameter_counts={name: value["trainable_parameters"] for name, value in plan["conditions"].items()})
    path = summary_path(config)
    write_json(path, summary)
    temporary = path.with_suffix(".txt.tmp")
    temporary.write_text(format_summary(summary))
    temporary.replace(path.with_suffix(".txt"))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for name in ("manifest", "feature-dir", "tag-list", "prevalence", "output-dir", "summary-dir"):
        parser.add_argument(f"--{name}", type=Path)
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
        config = config_from_args(args)
        result = run_experiment(config)
        if config.dry_run or config.validate_only:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(format_summary(result), end="")
            print(f"\nReadable report: {summary_path(config).with_suffix('.txt')}\nDetailed results: {summary_path(config)}")
    except (ValueError, OSError) as error:
        LOGGER.error("%s", error)
        return 2
    return 0

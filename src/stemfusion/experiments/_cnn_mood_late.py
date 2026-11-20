"""Plain CNN Jamendo late fusion through the existing Music2Emo fitter."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from stemfusion.configuration import JAMENDO_LATE_CONDITIONS, JAMENDO_MID_CONDITIONS, _UniqueKeyLoader, load_configuration
from stemfusion.experiments import _cnn_mood as mid
from stemfusion.features.musicnn import SOURCES
from stemfusion.training import music2emo
from stemfusion.training.late_fusion import branch_seed
from stemfusion.training.music2emo_late_fusion import assemble_ensemble
from stemfusion.training.reporting import file_checksum, fingerprint, reusable_artifacts, summarize_results, write_json

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs/experiments/mtg_jamendo/late_fusion_cnn.yaml"


@dataclass(frozen=True)
class JamendoCNNLateConfig(mid.JamendoCNNConfig):
    reference_output_dir: Path | None = None


def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    return fingerprint({"cnn_mid_implementation": mid.implementation_identity(),
                        **{name: file_checksum(root / name) for name in
                           ("experiments/_cnn_mood_late.py", "training/late_fusion.py",
                            "training/music2emo_late_fusion.py")}})


def config_from_args(args):
    path = args.config.resolve()
    raw = yaml.load(path.read_text(), Loader=_UniqueKeyLoader)
    if not isinstance(raw, dict) or set(raw) != {"protocol", "dataset", "features", "model", "paths"} or \
       raw["dataset"] != "mtg_jamendo" or raw["features"] != {
           "name": "msd_musicnn", "dimension_per_source": 200, "pooling": "track_window_mean", "harmony": False} or \
       raw["model"] != {"hidden_dimensions": [512, 256], "activation": "relu", "dropout": 0., "output_labels": 56}:
        raise ValueError("Expected Jamendo 200-d CNN features and a plain 512/256/56 ReLU head without dropout")
    defaults, provenance = load_configuration((path.parent / raw["protocol"]).resolve(), "jamendo_late_fusion")
    required = {"manifest", "feature_dir", "tag_list", "prevalence", "output_dir", "summary_dir", "reference_output_dir"}
    if set(raw["paths"]) != required:
        raise ValueError("Missing or unsupported Jamendo CNN late-fusion paths")
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
    return JamendoCNNLateConfig(**paths, **settings, conditions=defaults["runs"],
                               selected_run=args.run, config_provenance=provenance)


def branch_keys(condition):
    if condition["kind"] == "mixture_ensemble":
        return tuple(("mixture", index) for index in range(condition["matched_sources"]))
    return tuple((source, SOURCES.index(source)) for source in condition["sources"])


def branch_id(source, index):
    return f"mixture_{index + 1}" if source == "mixture" else source


def plan_experiment(config):
    # Retain the existing CNN mid configuration's seed and optimizer validation.
    plan = mid.plan_experiment(replace(config, conditions=JAMENDO_MID_CONDITIONS, selected_run=None))
    if not config.conditions or any(JAMENDO_LATE_CONDITIONS.get(name) != value for name, value in config.conditions.items()):
        raise ValueError("Use the fixed Jamendo late-fusion and standalone-stem conditions")
    if config.selected_run is not None and config.selected_run not in config.conditions:
        raise ValueError(f"Unknown condition: {config.selected_run}")
    if config.reference_output_dir is None:
        raise ValueError("Configure the completed CNN mid-fusion reference_output_dir")
    reference, output = config.reference_output_dir.resolve(), config.output_dir.resolve()
    if reference.is_relative_to(output) or output.is_relative_to(reference):
        raise ValueError("CNN reference and late-fusion outputs must have separate non-nested roots")
    conditions = {config.selected_run: config.conditions[config.selected_run]} if config.selected_run else config.conditions
    needed = {("mixture", 0)} | {key for value in conditions.values() for key in branch_keys(value)}
    for outer in config.seeds:
        if len({branch_seed(outer, index) for index in range(6)}) != 6:
            raise ValueError("Derived branch seeds collide within an ensemble")
    for _, index in needed:
        if len({branch_seed(outer, index) for outer in config.seeds}) != len(config.seeds):
            raise ValueError("Derived branch seeds collide across outer seeds")
    parameters = plan["conditions"]["mixture_only"]["trainable_parameters"]
    plan.update(method="late_fusion", conditions={name: {**value, "trainable_parameters": len(branch_keys(value)) * parameters}
                                                for name, value in conditions.items()},
                aggregation="sum_logits_then_sigmoid", branch_input_dimension=200,
                branch_trainable_parameters=parameters, branches_per_outer_seed=len(needed),
                new_branch_fits_per_seed=len(needed) - 1,
                maximum_new_branch_fits=(len(needed) - 1) * len(config.seeds),
                report_conditions_per_seed=len(conditions), reference_output_dir=str(reference),
                reference_mode="read_only_completed_cnn_mixture_only",
                branch_schedule={str(outer): [{"branch_id": branch_id(source, index), "source": source, "index": index,
                                               "actual_training_seed": branch_seed(outer, index)}
                                              for source, index in sorted(needed, key=lambda item: (item[0] != "mixture", item[1]))]
                                 for outer in config.seeds})
    return plan


def load_reference_branch(config, data, outer_seed):
    """Read CNN-M without invoking its fitter or changing its original identity."""
    directory = config.reference_output_dir / "mixture_only" / f"seed_{outer_seed}"
    marker_path = directory / "run_manifest.json"
    if not marker_path.is_file():
        raise ValueError(f"Completed CNN mixture-only reference required; no reference training will be started: {directory}")
    marker = json.loads(marker_path.read_text())
    required = {"best.pt", "result.json", "resolved_config.json", "predictions_validation.npz", "predictions_test.npz"}
    if not required <= marker.get("files", {}).keys() or not marker.get("signature"):
        raise ValueError(f"CNN reference completion record is missing required artifacts: {directory}")
    reusable_artifacts(directory, marker["signature"])
    result = json.loads((directory / "result.json").read_text())
    protocol = result["protocol"]
    expected = {
        "task": "jamendo_cnn_mid_fusion", "condition": "mixture_only", "source": ["mixture"],
        "inputs": 1, "kind": "reference", "feature_dimension_per_source": 200,
        "hidden_dimensions": [512, 256], "head_dropout": 0., "harmony_features": False,
        "sampling": "fixed_track_window_mean", "repeated_mixture_copies": 1,
        "epochs": config.epochs, "batch_size": config.batch_size, "learning_rate": config.learning_rate,
        "optimizer": "Adam", "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 0.,
        "early_stopping": {"monitor": "val_loss", "mode": "min", "patience": config.early_stopping_patience,
                           "min_delta": config.early_stopping_min_delta, "check_every_n_epochs": 1},
        "selection": "min_validation_weighted_bce", "ties": "earliest_epoch", "scheduler": "none",
        "save_top_k": 5, "save_last": True, "device": str(config.device), "num_workers": config.num_workers,
        "matmul_precision": "medium", "attention_backend": "math", "deterministic_algorithms": True,
        "threshold": "validation_macro_f1_global_0_to_1_step_0.01_ties_lowest", "accuracy": "labelwise",
    }
    for key, value in expected.items():
        if key not in protocol or fingerprint(protocol[key]) != fingerprint(value):
            raise ValueError(f"CNN mixture reference has incompatible {key}: {directory}")
    implementation = mid.implementation_identity()
    signature = fingerprint({"seed": outer_seed, "protocol": protocol,
                             "data": data.data_fingerprint, "implementation": implementation})
    if result.get("signature") != signature or marker["signature"] != signature or \
       result.get("implementation_sha256") != implementation or result.get("seed") != outer_seed or \
       result.get("data_fingerprint") != data.data_fingerprint or tuple(result.get("tag_names", ())) != data.tag_names:
        raise ValueError(f"CNN mixture reference settings, tag mapping, or feature cohort differ: {directory}")
    if Path(result.get("checkpoint", "")).resolve() != (directory / "best.pt").resolve() or \
       result.get("trainable_parameters") != mid.model_parameter_count(200, 56, (512, 256)):
        raise ValueError(f"CNN reference checkpoint or model parameter count differs: {directory}")
    for split in ("validation", "test"):
        records = data.splits[split]
        with np.load(directory / f"predictions_{split}.npz", allow_pickle=False) as saved:
            if saved["track_ids"].tolist() != [record.track_id for record in records] or \
               not np.array_equal(saved["targets"], np.stack([record.labels for record in records])):
                raise ValueError(f"CNN reference prediction IDs/order or labels differ: {directory}")
            if any(saved[name].shape != (len(records), 56) or not np.isfinite(saved[name]).all()
                   for name in ("logits", "probabilities")):
                raise ValueError(f"CNN reference predictions have invalid shapes or values: {directory}")
    return enrich_branch(result, data, outer_seed, "mixture", 0, directory)


def enrich_branch(result, data, outer_seed, source, index, directory):
    return {**result, "branch_id": branch_id(source, index), "source": source, "index": index,
            "outer_seed": outer_seed, "actual_training_seed": branch_seed(outer_seed, index),
            "cohort_fingerprint": data.data_fingerprint, "output_directory": str(directory)}


def train_branch(config, data, outer_seed, source, index, implementation):
    identifier, seed = branch_id(source, index), branch_seed(outer_seed, index)
    values = {item.name: getattr(config, item.name) for item in fields(mid.JamendoCNNConfig)}
    values["output_dir"] = config.output_dir / "branches" / identifier
    fit_config = mid.JamendoCNNConfig(**values)
    result = music2emo.fit_model(
        fit_config, data, seed, model_factory=lambda: mid.CNNMoodClassifier(1),
        dataset_factory=lambda records, training: mid.JamendoCNNDataset(records, sources=(source,)),
        implementation=implementation,
        protocol_updates={"task": "jamendo_cnn_late_branch", "condition": identifier, "source": [source],
                          "inputs": 1, "kind": "branch", "branch_id": identifier, "branch_index": index,
                          "outer_seed": outer_seed, "actual_training_seed": seed,
                          "feature_dimension_per_source": 200, "hidden_dimensions": [512, 256],
                          "head_dropout": 0., "harmony_features": False, "sampling": "fixed_track_window_mean",
                          "repeated_mixture_copies": 1,
                          "threshold": "validation_macro_f1_global_0_to_1_step_0.01_ties_lowest",
                          "accuracy": "labelwise"})
    if result["trainable_parameters"] != mid.model_parameter_count(200, 56, (512, 256)):
        raise ValueError("All CNN late-fusion branches must use the same 200/512/256/56 model")
    return enrich_branch(result, data, outer_seed, source, index, fit_config.output_dir / f"seed_{seed}")


def summary_path(config):
    name = "summary_late_fusion"
    if config.selected_run:
        name += "_run_" + config.selected_run
    if config.seeds != mid.SEEDS:
        name += "_seeds_" + "_".join(map(str, config.seeds))
    return config.summary_dir / f"{name}.json"


def format_summary(summary):
    def cell(value, scale=1):
        return f"{value['mean'] * scale:.4f}" + (f" +/- {value['sample_sd'] * scale:.4f}"
                                                if value["sample_sd"] is not None else " (SD unavailable)")
    def cells(metrics):
        return [cell(metrics[key], 100 if key == "accuracy" else 1)
                for key in ("accuracy", "f1_macro", "roc_auc_macro", "pr_auc_macro")]
    lines = ["MTG-Jamendo: frozen MSD_musicnn late fusion; plain 512/256 ReLU branch heads",
             "Seeds: " + ", ".join(map(str, summary["seeds"])) + "; mean +/- sample SD (ddof=1).",
             "Features: fixed track-window mean, 200 dimensions per source; no chord/key features.",
             "Checkpoints: minimum validation weighted BCE; ensembles sum raw logits, then sigmoid.",
             "Accuracy is label-wise. Each model uses its own global validation-macro-F1 threshold.",
             "Standalone stems reuse selected branch checkpoints; PR-AUC means macro average precision.", "",
             "Condition | Accuracy (%) | Macro F1 | Macro ROC-AUC | Macro PR-AUC",
             "--- | --- | --- | --- | ---"]
    for name, metrics in summary["conditions"].items():
        lines.append(" | ".join([name, *cells(metrics)]))
    if summary["paired_differences"]:
        lines += ["", "Paired differences: stem fusion minus its matched mixture ensemble.",
                  "Comparison | Accuracy (percentage points) | Macro F1 | Macro ROC-AUC | Macro PR-AUC",
                  "--- | --- | --- | --- | ---"]
        for name, comparison in summary["paired_differences"].items():
            lines.append(" | ".join([f"{name} - {comparison['control']}", *cells(comparison["metrics"])]))
    return "\n".join(lines) + "\n"


def run_experiment(config):
    plan = plan_experiment(config)
    if config.dry_run:
        return plan
    data = mid.load_jamendo_cnn_features(config.manifest, config.feature_dir, config.tag_list, config.prevalence)
    # Require all requested references before writing results or training branches.
    references = {seed: load_reference_branch(config, data, seed) for seed in config.seeds}
    if config.validate_only:
        return {"validate_only": True, "audit": data.audit, "plan": plan,
                "reference_seeds_verified": list(references)}
    implementation = implementation_identity()
    conditions = {name: config.conditions[name] for name in plan["conditions"]}
    needed = {key for value in conditions.values() for key in branch_keys(value)}
    results, registry = {name: {} for name in conditions}, {}
    for outer in config.seeds:
        branches = {("mixture", 0): references[outer]}
        for source, index in sorted(needed - branches.keys(), key=lambda item: (item[0] != "mixture", item[1])):
            branches[(source, index)] = train_branch(config, data, outer, source, index, implementation)
        registry[outer] = {value["branch_id"]: value for value in branches.values()}
        for name, condition in conditions.items():
            members = [branches[key] for key in branch_keys(condition)]
            if condition["kind"] in {"reference", "stem_only"}:
                result = {**members[0], "seed": outer, "condition": name, "additional_training_jobs": 0}
            else:
                result = assemble_ensemble(members, condition=name, outer_seed=outer,
                    output_directory=config.output_dir / "ensembles" / name / f"seed_{outer}",
                    prevalence=data.prevalence, implementation=implementation,
                    resume=config.resume, overwrite=config.overwrite)
            results[name][outer] = result
    comparisons = {name: f"mixture_ensemble_{len(value['sources'])}" for name, value in conditions.items()
                   if value["kind"] == "fusion"}
    summary = summarize_results(results, config.seeds, comparisons)
    summary.update(dataset="mtg_jamendo", method="late_fusion", features="msd_musicnn", runs=results, branches=registry,
                   plan=plan, data_audit=data.audit, accuracy_definition="labelwise",
                   threshold_policy="validation_macro_f1_global_grid", implementation_sha256=implementation,
                   config_provenance=config.config_provenance, reference_output_dir=str(config.reference_output_dir),
                   standalone_conditions=[name for name, value in conditions.items() if value["kind"] == "stem_only"],
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
    for name in ("manifest", "feature-dir", "tag-list", "prevalence", "output-dir", "summary-dir", "reference-output-dir"):
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

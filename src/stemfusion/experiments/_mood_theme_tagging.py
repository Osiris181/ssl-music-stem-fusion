"""Shared Jamendo experiment interface, scheduling, and reporting.

Like the genre experiment controller, this module handles arguments and both
fusion families. Model fitting remains in training/music2emo.py; multilabel
ensemble evaluation remains in training/music2emo_late_fusion.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from stemfusion.configuration import (
    JAMENDO_MID_CONDITIONS as MID_CONDITIONS, JAMENDO_LATE_CONDITIONS as LATE_CONDITIONS,
    YAMLArgumentParser, configuration_dict, configure_yaml,
)
from stemfusion.data.jamendo_mid_fusion import SOURCES, JamendoMidFusionDataset
from stemfusion.data.jamendo_late_fusion import prepare_late_fusion_data as prepare_cached_fusion_data
from stemfusion.experiments.mtg_jamendo import reproduction
from stemfusion.experiments.mtg_jamendo._checksum_compatibility import implementation_identity, execution_identity
from stemfusion.models.music2emo_mid_fusion import Music2EmoMidFusion, parameter_count
from stemfusion.training import music2emo
from stemfusion.training.late_fusion import branch_seed
from stemfusion.training.music2emo_late_fusion import assemble_ensemble
from stemfusion.training.reporting import (
    file_checksum, fingerprint, finish_artifacts, reusable_artifacts, summarize_results, write_json,
)

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class MidFusionConfig(reproduction.ReproductionConfig):
    reference_output_dir: Path | None = None
    conditions: dict = field(default_factory=lambda: json.loads(json.dumps(MID_CONDITIONS)))
    selected_run: str | None = None
    verify_checksums: bool = True



def _reference_config(config):
    if config.reference_output_dir is None:
        raise ValueError("Configure reference_output_dir explicitly; no checkpoint path is hardcoded in Python.")
    values = {item.name: getattr(config, item.name) for item in fields(reproduction.ReproductionConfig)}
    values.update(output_dir=config.reference_output_dir, resume=True, overwrite=False)
    return reproduction.ReproductionConfig(**values)


def _load_predictions(path, records):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ("track_ids", "targets", "logits", "probabilities")}
    if arrays["track_ids"].tolist() != [record.track_id for record in records]:
        raise ValueError("Reference prediction IDs/order differ from the current manifest.")
    if not np.array_equal(arrays["targets"], np.stack([record.labels for record in records])):
        raise ValueError("Reference prediction targets differ from the current manifest.")
    expected = (len(records), 56)
    if any(arrays[name].shape != expected or not np.isfinite(arrays[name]).all()
           for name in ("logits", "probabilities")):
        raise ValueError("Reference predictions have invalid shapes or non-finite values.")
    return arrays


def _reference_result(config, data, seed, implementation):
    reference = reproduction.fit_seed(_reference_config(config), data.mixture, seed)
    source = config.reference_output_dir / f"seed_{seed}"
    signature = fingerprint({"reference_signature": reference["signature"], "implementation": implementation,
                             "predictions": {split: file_checksum(source / f"predictions_{split}.npz")
                                             for split in ("validation", "test")}})
    directory = config.output_dir / "mixture_only" / f"seed_{seed}"
    if directory.exists() and not config.overwrite:
        if config.resume and reusable_artifacts(directory, signature):
            return json.loads((directory / "result.json").read_text())
        raise ValueError(f"Incomplete reference evaluation exists: {directory}; use --overwrite.")
    arrays = {split: _load_predictions(source / f"predictions_{split}.npz", data.mixture.splits[split])
              for split in ("validation", "test")}
    threshold = music2emo.select_threshold(arrays["validation"])
    result = {**reference, "signature": signature, "reference_signature": reference["signature"],
              "condition": "mixture_only", "threshold": threshold, "cohort_fingerprint": data.cohort_fingerprint,
              "reference_predictions": str(source), "mid_fusion_implementation_sha256": implementation}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_manifest.json").unlink(missing_ok=True)
    for split in ("validation", "test"):
        result[split] = {**reference[split], **music2emo.threshold_metrics(arrays[split], threshold)}
        arrays[split]["predictions"] = (arrays[split]["probabilities"] >= threshold).astype(np.uint8)
        arrays[split]["threshold"] = np.asarray(threshold)
        music2emo._save_predictions(directory / f"predictions_{split}.npz", arrays[split])
    write_json(directory / "result.json", result)
    finish_artifacts(directory, signature, ["result.json", "predictions_validation.npz", "predictions_test.npz"])
    return result


def _fit_condition(config, data, name, condition, seed, implementation):
    sources = tuple(condition["sources"])
    count = condition.get("matched_sources", len(sources))
    repeated = condition["kind"] == "repeated_mixture"
    values = {item.name: getattr(config, item.name) for item in fields(reproduction.ReproductionConfig)}
    values["output_dir"] = config.output_dir / name
    fit_config = reproduction.ReproductionConfig(**values)
    result = music2emo.fit_model(
        fit_config, data.condition(sources), seed,
        model_factory=lambda: Music2EmoMidFusion(count, repeated_mixture=repeated),
        dataset_factory=lambda records, training: JamendoMidFusionDataset(records, sources=sources, training=training),
        implementation=implementation,
        protocol_updates={"condition": name, "source": list(sources), "inputs": count,
                          "kind": condition["kind"], "harmony_encoder": "shared",
                          "sampling": "shared_epoch_seeded_conditional_uniform_consecutive_segment_mean",
                          "repeated_mixture": "encode_once_repeat_complete_vector" if repeated else None,
                          "threshold": "validation_macro_f1_global_0_to_1_step_0.01_ties_lowest",
                          "accuracy": "labelwise", "cohort_fingerprint": data.cohort_fingerprint})
    if result["trainable_parameters"] != parameter_count(count):
        raise ValueError("Actual trainable parameters differ from the matched-control plan.")
    return {**result, "condition": name, "cohort_fingerprint": data.cohort_fingerprint}


def format_mid_summary(summary):
    lines = ["MTG-Jamendo: MERT + source-specific chord/key mid-fusion",
             f"Seeds: {', '.join(map(str, summary['seeds']))}; mean +/- sample SD (ddof=1).",
             "Checkpoint: minimum validation weighted BCE; PR-AUC = macro average precision.",
             "Accuracy is label-wise, not exact-match/subset accuracy. One global threshold maximizes validation macro F1.",
             "Shared chord Transformer; repeated controls duplicate the same complete encoded mixture vector.", "",
             "Condition | Accuracy (%) | Macro F1 | Macro ROC-AUC | Macro PR-AUC",
             "--- | --- | --- | --- | ---"]
    def render(value, scale=1):
        mean = value["mean"] * scale
        sd = value["sample_sd"]
        return f"{mean:.4f}" + (f" +/- {sd * scale:.4f}" if sd is not None else " (SD unavailable: one seed)")
    for name, metrics in summary["conditions"].items():
        cells = [render(metrics[key], 100 if key == "accuracy" else 1)
                 for key in ("accuracy", "f1_macro", "roc_auc_macro", "pr_auc_macro")]
        lines.append(" | ".join([name, *cells]))
    if summary["paired_differences"]:
        lines.extend(["", "Paired differences: fusion minus its matched repeated-mixture control.",
                      "Comparison | Accuracy (percentage points) | Macro F1 | Macro ROC-AUC | Macro PR-AUC",
                      "--- | --- | --- | --- | ---"])
        for name, comparison in summary["paired_differences"].items():
            cells = [render(comparison["metrics"][key], 100 if key == "accuracy" else 1)
                     for key in ("accuracy", "f1_macro", "roc_auc_macro", "pr_auc_macro")]
            lines.append(" | ".join([f"{name} - {comparison['control']}", *cells]))
    return "\n".join(lines) + "\n"


def run_mid_fusion(config: MidFusionConfig):
    if type(config.verify_checksums) is not bool:
        raise ValueError("verify_checksums must be a boolean.")
    if (not config.seeds or len(set(config.seeds)) != len(config.seeds) or
            any(type(seed) is not int or not 0 <= seed < 2**32 for seed in config.seeds)):
        raise ValueError("Choose distinct valid integer training seeds.")
    if (config.epochs <= 0 or config.batch_size <= 0 or config.num_workers < 0 or
            not math.isfinite(config.learning_rate) or config.learning_rate <= 0 or
            type(config.early_stopping_patience) is not int or config.early_stopping_patience < 0 or
            not math.isfinite(config.early_stopping_min_delta) or config.early_stopping_min_delta < 0):
        raise ValueError("Invalid training/stopping settings.")
    if not config.conditions or any(MID_CONDITIONS.get(name) != value for name, value in config.conditions.items()):
        raise ValueError("Use the fixed Jamendo mid-fusion conditions.")
    if config.selected_run is not None and config.selected_run not in config.conditions:
        raise ValueError(f"Unknown condition: {config.selected_run}")
    if config.reference_output_dir is None or any(
            config.reference_output_dir.resolve() == (config.output_dir / name).resolve() for name in MID_CONDITIONS):
        raise ValueError("Reference checkpoints need a separate configured directory from condition outputs.")
    conditions = ({config.selected_run: config.conditions[config.selected_run]} if config.selected_run else config.conditions)
    selected_sources = tuple(source for source in SOURCES if any(source in value["sources"] for value in conditions.values()))
    plan = {"conditions": conditions, "seeds": list(config.seeds), "sources": list(selected_sources),
            "epochs": config.epochs, "batch_size": config.batch_size, "checkpoint": "minimum_validation_weighted_bce",
            "harmony_encoder": "shared", "reference_output_dir": str(config.reference_output_dir),
            "verify_checksums": config.verify_checksums}
    if config.dry_run:
        LOGGER.info("Jamendo mid-fusion plan (no features/model loaded): %s", json.dumps(plan, sort_keys=True))
        return {"dry_run": True, **plan}
    if not config.verify_checksums:
        LOGGER.warning("Feature payload checksum verification is disabled; trusting cache completion markers. "
                       "Finite changes to cached values may go undetected, including during checkpoint reuse. "
                       "ID/split/shape/finiteness/settings/alignment checks remain mandatory.")
    data = prepare_cached_fusion_data(config.manifest, config.mert_dir, config.harmony_dir,
                                      config.tag_list, config.author_metadata, selected_sources,
                                      verify_checksums=config.verify_checksums)
    LOGGER.info("Loaded source-aligned cached inputs: %s", json.dumps(data.audit, sort_keys=True))
    if config.validate_only:
        return {"validate_only": True, "audit": data.audit, **plan}
    implementation = implementation_identity("mid")
    results = {name: {} for name in conditions}
    for seed in config.seeds:
        for name, condition in conditions.items():
            result = (_reference_result(config, data, seed, implementation) if condition["kind"] == "reference"
                      else _fit_condition(config, data, name, condition, seed, implementation))
            results[name][seed] = result
    comparisons = {name: f"repeated_mixture_{len(condition['sources'])}" for name, condition in conditions.items()
                   if condition["kind"] == "fusion"}
    cohort_results = {name: {seed: {**result, "input_fingerprint": result["data_fingerprint"],
                                   "data_fingerprint": data.cohort_fingerprint} for seed, result in runs.items()}
                      for name, runs in results.items()}
    summary = summarize_results(cohort_results, config.seeds, comparisons)
    summary.update(dataset="mtg_jamendo", method="mid_fusion", runs=results, plan=plan, data_audit=data.audit,
                   accuracy_definition="labelwise", threshold_policy="validation_macro_f1_global_grid",
                   implementation_sha256=implementation, config_provenance=config.config_provenance,
                   execution_implementation_sha256=execution_identity("mid"),
                   parameter_counts={name: parameter_count(value.get("matched_sources", len(value["sources"])))
                                     for name, value in conditions.items()})
    name = "summary_mid_fusion"
    if config.selected_run:
        name += "_run_" + config.selected_run
    if config.seeds != reproduction.SEEDS:
        name += "_seeds_" + "_".join(map(str, config.seeds))
    write_json(config.summary_dir / f"{name}.json", summary)
    temporary = config.summary_dir / f".{name}.txt.tmp"
    temporary.write_text(format_mid_summary(summary))
    temporary.replace(config.summary_dir / f"{name}.txt")
    print(format_mid_summary(summary), end="")
    return summary


@dataclass(frozen=True)
class LateFusionConfig(reproduction.ReproductionConfig):
    reference_output_dir: Path | None = None
    conditions: dict = field(default_factory=lambda: json.loads(json.dumps(LATE_CONDITIONS)))
    selected_run: str | None = None
    verify_checksums: bool = True



def branch_keys(condition):
    if condition["kind"] == "mixture_ensemble":
        return tuple(("mixture", index) for index in range(condition["matched_sources"]))
    return tuple((source, SOURCES.index(source)) for source in condition["sources"])


def branch_id(source, index):
    if source == "mixture":
        if type(index) is not int or not 0 <= index <= 5:
            raise ValueError("Mixture branches use positions zero through five.")
        return f"mixture_{index + 1}"
    if source not in SOURCES or index != SOURCES.index(source):
        raise ValueError("Stem branches use their fixed source-order positions.")
    return source


def _reference_branch(config, data, outer_seed, implementation):
    reference = reproduction.fit_seed(_reference_config(config), data.mixture, outer_seed)
    original = config.reference_output_dir / f"seed_{outer_seed}"
    directory = config.output_dir / "branches/mixture_1" / f"seed_{outer_seed}"
    signature = fingerprint({"reference_signature": reference["signature"], "implementation": implementation,
                             "cohort_fingerprint": data.cohort_fingerprint,
                             "predictions": {split: file_checksum(original / f"predictions_{split}.npz")
                                             for split in ("validation", "test")}})
    if directory.exists() and not config.overwrite:
        if config.resume and reusable_artifacts(directory, signature):
            return json.loads((directory / "result.json").read_text())
        raise ValueError(f"Incomplete reference branch evaluation: {directory}; use --overwrite.")
    arrays = {split: _load_predictions(original / f"predictions_{split}.npz", data.mixture.splits[split])
              for split in ("validation", "test")}
    threshold = music2emo.select_threshold(arrays["validation"])
    result = {**reference, "signature": signature, "reference_signature": reference["signature"],
              "branch_id": "mixture_1", "source": "mixture", "index": 0, "outer_seed": outer_seed,
              "actual_training_seed": outer_seed, "cohort_fingerprint": data.cohort_fingerprint,
              "threshold": threshold, "output_directory": str(directory),
              "late_fusion_implementation_sha256": implementation, "reference_directory": str(original)}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_manifest.json").unlink(missing_ok=True)
    for split in ("validation", "test"):
        result[split] = {**reference[split], **music2emo.threshold_metrics(arrays[split], threshold)}
        arrays[split]["predictions"] = (arrays[split]["probabilities"] >= threshold).astype(np.uint8)
        arrays[split]["threshold"] = np.asarray(threshold)
        music2emo._save_predictions(directory / f"predictions_{split}.npz", arrays[split])
    write_json(directory / "resolved_config.json", configuration_dict(config))
    write_json(directory / "result.json", result)
    finish_artifacts(directory, signature, ["result.json", "resolved_config.json", "predictions_validation.npz", "predictions_test.npz"])
    return result


def train_branch(config, data, *, outer_seed, source, index, implementation):
    identifier = branch_id(source, index)
    seed = branch_seed(outer_seed, index)
    if source == "mixture" and index == 0:
        return _reference_branch(config, data, outer_seed, implementation)
    values = {item.name: getattr(config, item.name) for item in fields(reproduction.ReproductionConfig)}
    values["output_dir"] = config.output_dir / "branches" / identifier
    fit_config = reproduction.ReproductionConfig(**values)
    sources = (source,)
    result = music2emo.fit_model(fit_config, data.condition(sources), seed,
        model_factory=lambda: Music2EmoMidFusion(1),
        dataset_factory=lambda records, training: JamendoMidFusionDataset(records, sources=sources, training=training),
        implementation=implementation,
        protocol_updates={"task": "jamendo_late_branch", "condition": identifier, "source": [source],
                          "branch_id": identifier, "branch_index": index, "outer_seed": outer_seed,
                          "actual_training_seed": seed, "harmony_encoder": "independent_per_branch",
                          "threshold": "validation_macro_f1_global_0_to_1_step_0.01_ties_lowest",
                          "accuracy": "labelwise", "cohort_fingerprint": data.cohort_fingerprint})
    if result["trainable_parameters"] != parameter_count(1):
        raise ValueError("All late-fusion branches must retain the same Music2Emo architecture.")
    return {**result, "branch_id": identifier, "source": source, "index": index, "outer_seed": outer_seed,
            "actual_training_seed": seed, "cohort_fingerprint": data.cohort_fingerprint,
            "output_directory": str(fit_config.output_dir / f"seed_{seed}")}


def _standalone_result(branch, name):
    return {**branch, "seed": branch["outer_seed"], "condition": name,
            "input_fingerprint": branch["data_fingerprint"], "data_fingerprint": branch["cohort_fingerprint"],
            "standalone_branch": branch["branch_id"], "additional_training_jobs": 0}


def format_late_summary(summary):
    lines = ["MTG-Jamendo: MERT + source-specific chord/key unweighted late fusion",
             f"Seeds: {', '.join(map(str, summary['seeds']))}; mean +/- sample SD (ddof=1).",
             "Each branch: minimum validation weighted BCE checkpoint; ensembles sum raw logits, then sigmoid.",
             "Accuracy is label-wise. A global validation-macro-F1 threshold is frozen for each reported model.",
             "PR-AUC denotes macro average precision. Stem-only rows reuse the selected stem branches.", ""]
    def render(value, scale=1):
        sd = value["sample_sd"]
        return f"{value['mean'] * scale:.4f}" + (f" +/- {sd * scale:.4f}" if sd is not None else " (SD unavailable: one seed)")
    for title, standalone in (("Mixture reference and late-fusion ensembles", False), ("Supporting stem-only results", True)):
        selected = [(name, value) for name, value in summary["conditions"].items()
                    if (LATE_CONDITIONS[name]["kind"] == "stem_only") == standalone]
        if not selected:
            continue
        lines.extend([title, "Condition | Accuracy (%) | Macro F1 | Macro ROC-AUC | Macro PR-AUC",
                      "--- | --- | --- | --- | ---"])
        for name, metrics in selected:
            cells = [render(metrics[key], 100 if key == "accuracy" else 1)
                     for key in ("accuracy", "f1_macro", "roc_auc_macro", "pr_auc_macro")]
            lines.append(" | ".join([name, *cells]))
        lines.append("")
    if summary["paired_differences"]:
        lines.extend(["Paired differences: stem fusion minus its matched mixture ensemble.",
                      "Comparison | Accuracy (percentage points) | Macro F1 | Macro ROC-AUC | Macro PR-AUC",
                      "--- | --- | --- | --- | ---"])
        for name, comparison in summary["paired_differences"].items():
            cells = [render(comparison["metrics"][key], 100 if key == "accuracy" else 1)
                     for key in ("accuracy", "f1_macro", "roc_auc_macro", "pr_auc_macro")]
            lines.append(" | ".join([f"{name} - {comparison['control']}", *cells]))
    return "\n".join(lines) + "\n"


def run_late_fusion(config: LateFusionConfig):
    if type(config.verify_checksums) is not bool:
        raise ValueError("verify_checksums must be a boolean.")
    if (not config.seeds or len(set(config.seeds)) != len(config.seeds) or
            any(type(seed) is not int or not 0 <= seed < 2**32 for seed in config.seeds)):
        raise ValueError("Choose distinct valid integer outer seeds.")
    if (config.epochs <= 0 or config.batch_size <= 0 or config.num_workers < 0 or
            not math.isfinite(config.learning_rate) or config.learning_rate <= 0 or
            type(config.early_stopping_patience) is not int or config.early_stopping_patience < 0 or
            not math.isfinite(config.early_stopping_min_delta) or config.early_stopping_min_delta < 0):
        raise ValueError("Invalid training/stopping settings.")
    if not config.conditions or any(LATE_CONDITIONS.get(name) != value for name, value in config.conditions.items()):
        raise ValueError("Use the fixed Jamendo late-fusion/stem-only report conditions.")
    if config.selected_run is not None and config.selected_run not in config.conditions:
        raise ValueError(f"Unknown condition: {config.selected_run}")
    if config.reference_output_dir is None or any((config.output_dir / child).resolve().is_relative_to(
            config.reference_output_dir.resolve()) for child in ("branches", "ensembles")):
        raise ValueError("Reference checkpoints must be separate from new branch/ensemble outputs.")
    conditions = ({config.selected_run: config.conditions[config.selected_run]} if config.selected_run else config.conditions)
    needed = sorted({key for value in conditions.values() for key in branch_keys(value)},
                    key=lambda item: (item[0] != "mixture", item[1]))
    for outer in config.seeds:
        if len({branch_seed(outer, index) for index in range(6)}) != 6:
            raise ValueError("Derived position seeds collide within an ensemble.")
    for source, index in needed:
        if len({branch_seed(outer, index) for outer in config.seeds}) != len(config.seeds):
            raise ValueError("Derived branch seeds collide across outer runs.")
    sources = tuple(source for source in SOURCES if source == "mixture" or any(key[0] == source for key in needed))
    plan = {"conditions": conditions, "sources": list(sources), "seeds": list(config.seeds),
            "branches_per_outer_seed": len(needed), "maximum_training_jobs": len(needed) * len(config.seeds),
            "report_conditions_per_seed": len(conditions), "aggregation": "sum_logits_then_sigmoid",
            "branch_seed_rule": "index 0 = outer seed; otherwise first 32 SHA256 bits of stemfusion-phase2:<outer>:<index>",
            "branch_schedule": {str(outer): [{"branch_id": branch_id(source, index), "source": source, "index": index,
                                               "actual_training_seed": branch_seed(outer, index)} for source, index in needed]
                                for outer in config.seeds},
            "epochs": config.epochs, "batch_size": config.batch_size,
            "reference_output_dir": str(config.reference_output_dir), "verify_checksums": config.verify_checksums}
    if config.dry_run:
        LOGGER.info("Jamendo late-fusion plan (no features/model loaded): %s", json.dumps(plan, sort_keys=True))
        return {"dry_run": True, **plan}
    if not config.verify_checksums:
        LOGGER.warning("Feature payload checksum verification is disabled; trusting cache completion markers. "
                       "Finite changes to cached values may go undetected, including during checkpoint reuse. "
                       "ID/split/shape/finiteness/settings/alignment checks remain mandatory.")
    data = prepare_cached_fusion_data(config.manifest, config.mert_dir, config.harmony_dir,
                                    config.tag_list, config.author_metadata, sources,
                                    verify_checksums=config.verify_checksums)
    LOGGER.info("Loaded cached inputs for independent branches: %s", json.dumps(data.audit, sort_keys=True))
    if config.validate_only:
        return {"validate_only": True, "audit": data.audit, **plan}
    implementation = implementation_identity("late")
    results = {name: {} for name in conditions}
    registry = {}
    for outer in config.seeds:
        branches = {key: train_branch(config, data, outer_seed=outer, source=key[0], index=key[1], implementation=implementation)
                    for key in needed}
        registry[outer] = {branch["branch_id"]: branch for branch in branches.values()}
        for name, condition in conditions.items():
            members = [branches[key] for key in branch_keys(condition)]
            if condition["kind"] in {"reference", "stem_only"}:
                result = _standalone_result(members[0], name)
            else:
                result = assemble_ensemble(members, condition=name, outer_seed=outer,
                    output_directory=config.output_dir / "ensembles" / name / f"seed_{outer}",
                    prevalence=data.mixture.prevalence, implementation=implementation,
                    resume=config.resume, overwrite=config.overwrite)
            results[name][outer] = result
    comparisons = {name: f"mixture_ensemble_{len(value['sources'])}" for name, value in conditions.items()
                   if value["kind"] == "fusion"}
    summary = summarize_results(results, config.seeds, comparisons)
    summary.update(dataset="mtg_jamendo", method="late_fusion", runs=results, branches=registry,
                   plan=plan, data_audit=data.audit, accuracy_definition="labelwise",
                   threshold_policy="validation_macro_f1_global_grid", implementation_sha256=implementation,
                   execution_implementation_sha256=execution_identity("late"),
                   config_provenance=config.config_provenance,
                   standalone_conditions=[name for name, value in conditions.items() if value["kind"] == "stem_only"])
    name = "summary_late_fusion"
    if config.selected_run:
        name += "_run_" + config.selected_run
    if config.seeds != reproduction.SEEDS:
        name += "_seeds_" + "_".join(map(str, config.seeds))
    write_json(config.summary_dir / f"{name}.json", summary)
    temporary = config.summary_dir / f".{name}.txt.tmp"
    temporary.write_text(format_late_summary(summary))
    temporary.replace(config.summary_dir / f"{name}.txt")
    print(format_late_summary(summary), end="")
    return summary


def configure_parser(parser, family):
    if family not in {"mid_fusion", "late_fusion"}:
        raise ValueError("Unknown Jamendo fusion family.")
    parser.description = f"Run the MTG-Jamendo {family.replace('_', '-')} matrix with matched controls and fixed outer seeds."
    for name in ("manifest", "mert-dir", "harmony-dir", "tag-list", "author-metadata", "output-dir", "summary-dir", "reference-output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", dest="seeds", type=int, action="append",
                        help="Repeat for a seed subset; replaces the YAML seed list.")
    for name, kind in (("epochs", int), ("batch-size", int), ("learning-rate", float),
                       ("early-stopping-patience", int), ("early-stopping-min-delta", float), ("num-workers", int)):
        parser.add_argument(f"--{name}", type=kind)
    parser.add_argument("--device")
    parser.add_argument("--verify-checksums", action=argparse.BooleanOptionalAction,
                        help="Verify cached feature payload SHA256s (default: enabled). "
                             "--no-verify-checksums trusts local completion markers; structural checks remain mandatory.")
    for name in ("resume", "overwrite", "validate-only", "dry-run"):
        parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction)
    configure_yaml(parser, f"jamendo_{family}", f"experiments/mtg_jamendo/{family}.yaml")
    parser.set_defaults(fusion_family=family, command_handler=run_from_args)


def config_from_args(args):
    paths = {name: getattr(args, name).expanduser().resolve() for name in (
        "manifest", "mert_dir", "harmony_dir", "tag_list", "author_metadata", "output_dir", "summary_dir", "reference_output_dir")}
    explicit = args.yaml_provenance["cli_overrides"]
    if "output_dir" in explicit:
        if "summary_dir" not in explicit:
            paths["summary_dir"] = paths["output_dir"]
        if "reference_output_dir" not in explicit:
            paths["reference_output_dir"] = paths["output_dir"] / "reference"
    cls = {"mid_fusion": MidFusionConfig, "late_fusion": LateFusionConfig}[args.fusion_family]
    return cls(**paths, seeds=tuple(args.seeds), epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta, device=args.device, num_workers=args.num_workers,
        resume=args.resume, overwrite=args.overwrite, validate_only=args.validate_only, dry_run=args.dry_run,
        conditions=args.runs, selected_run=args.run, config_provenance=args.yaml_provenance,
        verify_checksums=args.verify_checksums)


def run_experiment(config):
    if isinstance(config, MidFusionConfig):
        return run_mid_fusion(config)
    if isinstance(config, LateFusionConfig):
        return run_late_fusion(config)
    raise TypeError("Use a Jamendo mid- or late-fusion configuration.")


def run_from_args(args):
    try:
        run_experiment(config_from_args(args))
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        LOGGER.error("Jamendo %s: %s", args.fusion_family.replace("_", "-"), error)
        return 2
    return 0


def direct_main(family, argv=None):
    parser = YAMLArgumentParser(prog=f"train_{'mid' if family == 'mid_fusion' else 'late'}.py")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    configure_parser(parser, family)
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    return run_from_args(args)

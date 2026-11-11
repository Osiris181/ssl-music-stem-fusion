"""Emotion late-fusion scheduling; reuse the established fitter and binary task.

The completed mid-fusion controller remains unchanged so its scientific run
identities and mixture-only checkpoints stay reusable without migrations.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from stemfusion.configuration import JAMENDO_LATE_CONDITIONS as LATE_CONDITIONS, YAMLArgumentParser, configure_yaml
from stemfusion.data.emotion_mid_fusion import LABEL_POLICY, TARGETS, prepare_emotion_features
from stemfusion.data.jamendo_mid_fusion import SOURCES, JamendoMidFusionDataset
from stemfusion.experiments import _emotion_recognition as mid
from stemfusion.experiments._mood_theme_tagging import branch_id, branch_keys
from stemfusion.models.music2emo_emotion import Music2EmoEmotionMidFusion, parameter_count
from stemfusion.training import music2emo
from stemfusion.training.emotion import BinaryEmotionTask
from stemfusion.training.emotion_late_fusion import BinaryEmotionLateTask, assemble_ensemble
from stemfusion.training.late_fusion import branch_seed
from stemfusion.training.reporting import file_checksum, fingerprint, summarize_results, write_json

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmotionLateFusionConfig(mid.EmotionMidFusionConfig):
    reference_output_dir: Path | None = None
    conditions: dict = field(default_factory=lambda: json.loads(json.dumps(LATE_CONDITIONS)))


def implementation_identity():
    package = Path(__file__).resolve().parents[1]
    names = ("experiments/_emotion_late_fusion.py", "experiments/_mood_theme_tagging.py",
             "training/emotion_late_fusion.py", "training/music2emo_late_fusion.py", "training/late_fusion.py")
    return fingerprint({"mid_fusion_dependencies": mid.implementation_identity(),
                        "late_conditions": LATE_CONDITIONS,
                        **{name: file_checksum(package / name) for name in names}})


def _plan(config):
    # Reuse the existing target, seed and optimizer/stopping validation unchanged.
    plan = mid._plan(replace(config, conditions=mid.MID_CONDITIONS, selected_run=None))
    if not config.conditions or any(LATE_CONDITIONS.get(name) != value for name, value in config.conditions.items()):
        raise ValueError("Use the shared fixed late-fusion/stem-only report conditions.")
    if config.selected_run is not None and config.selected_run not in config.conditions:
        raise ValueError(f"Unknown condition: {config.selected_run}")
    if config.reference_output_dir is None:
        raise ValueError("Configure reference_output_dir explicitly; no checkpoint path is hardcoded in Python.")
    reference, output = config.reference_output_dir.resolve(), config.output_dir.resolve()
    if output.is_relative_to(reference) or reference.is_relative_to(output):
        raise ValueError("Reference checkpoints and late-fusion outputs must have separate non-nested roots.")
    conditions = {config.selected_run: config.conditions[config.selected_run]} if config.selected_run else config.conditions
    needed = sorted({key for value in conditions.values() for key in branch_keys(value)},
                    key=lambda item: (item[0] != "mixture", item[1]))
    for outer in config.seeds:
        if len({branch_seed(outer, index) for index in range(6)}) != 6:
            raise ValueError("Derived position seeds collide within an ensemble.")
    for _, index in needed:
        if len({branch_seed(outer, index) for outer in config.seeds}) != len(config.seeds):
            raise ValueError("Derived branch seeds collide across outer runs.")
    sources = [source for source in SOURCES if source == "mixture" or any(key[0] == source for key in needed)]
    repetitions = len(config.targets) * len(config.seeds)
    references = int(("mixture", 0) in needed)
    plan.update(method="late_fusion", conditions=conditions, sources=sources,
                reference_output_dir=str(config.reference_output_dir), branches_per_target_outer_seed=len(needed),
                maximum_training_jobs=len(needed) * repetitions,
                maximum_new_branch_training_jobs=(len(needed) - references) * repetitions,
                maximum_reference_training_jobs=references * repetitions,
                report_conditions_per_target_seed=len(conditions), aggregation="sum_logits_then_sigmoid",
                branch_seed_rule="index 0 = outer seed; otherwise first 32 SHA256 bits of stemfusion-phase2:<outer>:<index>",
                branch_schedule={str(outer): [{"branch_id": branch_id(source, index), "source": source, "index": index,
                                              "actual_training_seed": branch_seed(outer, index)} for source, index in needed]
                                 for outer in config.seeds})
    return plan, needed


def train_branch(config, data, *, target, outer_seed, source, index, implementation):
    identifier, seed = branch_id(source, index), branch_seed(outer_seed, index)
    values = {item.name: getattr(config, item.name) for item in fields(mid.EmotionMidFusionConfig)}
    values.update(targets=(target,))
    sources = (source,)
    reference_branch = source == "mixture" and index == 0
    if reference_branch:
        # Reconstruct exactly the existing mixture-only fit, including protocol
        # and implementation. A compatible completion returns before model setup.
        # --overwrite affects only late outputs, never an existing mid reference.
        values.update(output_dir=config.reference_output_dir / target / "mixture_only", resume=True, overwrite=False)
        fit_implementation = mid.implementation_identity()
        protocol = {"condition": "mixture_only", "source": ["mixture"], "inputs": 1, "kind": "reference",
                    "label_policy": LABEL_POLICY, "harmony_encoder": "shared", "cohort_fingerprint": data.cohort_fingerprint,
                    "sampling": "shared_epoch_seeded_conditional_uniform_consecutive_segment_mean", "repeated_mixture": None}
    else:
        values.update(output_dir=config.output_dir / target / "branches" / identifier)
        fit_implementation = implementation
        protocol = {"fusion_family": "late_fusion", "condition": identifier, "source": [source],
                    "branch_id": identifier, "branch_index": index, "outer_seed": outer_seed,
                    "actual_training_seed": seed, "inputs": 1, "kind": "branch", "label_policy": LABEL_POLICY,
                    "harmony_encoder": "independent_per_branch", "cohort_fingerprint": data.cohort_fingerprint}
    fit_config = mid.EmotionMidFusionConfig(**values)
    LOGGER.info("PMEmo %s | outer seed %d | branch %s | training seed %d", target, outer_seed, identifier, seed)
    result = music2emo.fit_model(fit_config, data.condition(sources), seed,
        model_factory=lambda: Music2EmoEmotionMidFusion(1),
        dataset_factory=lambda records, training: JamendoMidFusionDataset(records, sources=sources, training=training),
        implementation=fit_implementation, protocol_updates=protocol,
        task=BinaryEmotionTask(target) if reference_branch else BinaryEmotionLateTask(target))
    if result["trainable_parameters"] != parameter_count(1):
        raise ValueError("All late-fusion branches must retain the same binary Music2Emo architecture.")
    return {**result, "branch_id": identifier, "source": source, "index": index, "outer_seed": outer_seed,
            "actual_training_seed": seed, "target": target, "cohort_fingerprint": data.cohort_fingerprint,
            "output_directory": str(fit_config.output_dir / f"seed_{seed}")}


def format_late_summary(summary):
    lines = [f"PMEmo: binary {summary['target']} | MERT + source-specific chord/key unweighted late fusion",
             f"Seeds: {', '.join(map(str, summary['seeds']))}; mean +/- sample SD (ddof=1).",
             "Ground truth: original 0-1 score >= 0.5 is high; score < 0.5 is low.",
             "Each branch: minimum validation weighted BCE checkpoint; ensembles sum raw logits, then sigmoid.",
             "Threshold: validation macro-F1 grid, ties to lowest; frozen for test.",
             "Accuracy is track-level; macro metrics average low/high. PR-AUC means average precision.",
             "Mixture reference reuses mid fusion; stem-only rows reuse the selected stem branches.", ""]
    def render(value, scale=1):
        sd = value["sample_sd"]
        return f"{value['mean'] * scale:.4f}" + (f" +/- {sd * scale:.4f}" if sd is not None else " (SD unavailable: one seed)")
    def cells(metrics):
        return [render(metrics[key], 100 if key == "accuracy" else 1)
                for key in ("accuracy", "f1_macro", "roc_auc_macro", "pr_auc_macro")]
    for title, standalone in (("Mixture reference and late-fusion ensembles", False), ("Supporting stem-only results", True)):
        rows = [(name, metrics) for name, metrics in summary["conditions"].items()
                if (LATE_CONDITIONS[name]["kind"] == "stem_only") == standalone]
        if rows:
            lines.extend([title, "Condition | Accuracy (%) | Macro F1 | Macro ROC-AUC | Macro PR-AUC",
                          "--- | --- | --- | --- | ---"])
            lines.extend(" | ".join([name, *cells(metrics)]) for name, metrics in rows)
            lines.append("")
    if summary["paired_differences"]:
        lines.extend(["Paired differences: stem fusion minus its matched mixture ensemble.",
                      "Comparison | Accuracy (percentage points) | Macro F1 | Macro ROC-AUC | Macro PR-AUC",
                      "--- | --- | --- | --- | ---"])
        for name, comparison in summary["paired_differences"].items():
            lines.append(" | ".join([f"{name} - {comparison['control']}", *cells(comparison["metrics"])]))
    return "\n".join(lines) + "\n"


def run_late_fusion(config, *, features=None):
    plan, needed = _plan(config)
    if config.dry_run:
        print(json.dumps({"dry_run": True, **plan}, indent=2, sort_keys=True))
        return {"dry_run": True, **plan}
    if not config.verify_checksums:
        LOGGER.warning("Feature payload checksums are disabled; trusting completion markers. Finite cache changes may go undetected. "
                       "ID/split/shape/finiteness/settings/alignment checks remain mandatory.")
    if features is None:
        features = prepare_emotion_features(config.manifest, config.mert_dir, config.harmony_dir,
            config.chord_root, config.chord_attr, tuple(plan["sources"]), verify_checksums=config.verify_checksums)
    target_data = {target: features.target(target) for target in config.targets}
    if config.validate_only:
        result = {"validate_only": True, "plan": plan, "targets": {target: data.audit for target, data in target_data.items()}}
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    implementation = implementation_identity()
    summaries = {}
    for target, data in target_data.items():
        results, registry = {name: {} for name in plan["conditions"]}, {}
        for outer in config.seeds:
            branches = {key: train_branch(config, data, target=target, outer_seed=outer, source=key[0], index=key[1],
                                           implementation=implementation) for key in needed}
            registry[outer] = {branch["branch_id"]: branch for branch in branches.values()}
            for name, condition in plan["conditions"].items():
                members = [branches[key] for key in branch_keys(condition)]
                if condition["kind"] in {"reference", "stem_only"}:
                    branch = members[0]
                    result = {**branch, "seed": outer, "condition": name, "input_fingerprint": branch["data_fingerprint"],
                              "data_fingerprint": data.cohort_fingerprint, "standalone_branch": branch["branch_id"],
                              "additional_training_jobs": 0}
                else:
                    result = assemble_ensemble(members, condition=name, outer_seed=outer, target=target,
                        output_directory=config.output_dir / target / "ensembles" / name / f"seed_{outer}",
                        prevalence=data.mixture.prevalence, implementation=implementation,
                        resume=config.resume, overwrite=config.overwrite)
                results[name][outer] = result
        comparisons = {name: f"mixture_ensemble_{len(value['sources'])}" for name, value in plan["conditions"].items()
                       if value["kind"] == "fusion"}
        summary = summarize_results(results, config.seeds, comparisons)
        summary.update(dataset="pmemo", target=target, method="late_fusion", runs=results, branches=registry,
                       plan=plan, data_audit=data.audit, label_policy=LABEL_POLICY, metric_conventions=BinaryEmotionLateTask(target).protocol(),
                       implementation_sha256=implementation, config_provenance=config.config_provenance,
                       standalone_conditions=[name for name, value in plan["conditions"].items() if value["kind"] == "stem_only"])
        name = f"summary_late_fusion_{target}"
        if config.selected_run:
            name += "_run_" + config.selected_run
        if config.seeds != mid.SEEDS:
            name += "_seeds_" + "_".join(map(str, config.seeds))
        path = config.summary_dir / name
        write_json(path.with_suffix(".json"), summary)
        temporary = path.with_name(f".{name}.txt.tmp")
        try:
            temporary.write_text(format_late_summary(summary), encoding="utf-8")
            temporary.replace(path.with_suffix(".txt"))
        finally:
            temporary.unlink(missing_ok=True)
        print(format_late_summary(summary), end="")
        LOGGER.info("Saved results: %s.{json,txt}", path)
        summaries[target] = summary
    return {"targets": summaries, "plan": plan}


def configure_parser(parser):
    for name in ("manifest", "mert-dir", "harmony-dir", "chord-root", "chord-attr", "output-dir", "summary-dir", "reference-output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, action="append", dest="targets", help="Repeat to select targets; default: both.")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    for name in ("epochs", "batch-size", "num-workers", "early-stopping-patience"):
        parser.add_argument(f"--{name}", type=int)
    for name in ("learning-rate", "early-stopping-min-delta"):
        parser.add_argument(f"--{name}", type=float)
    parser.add_argument("--device")
    for name in ("resume", "overwrite", "verify-checksums", "validate-only", "dry-run"):
        parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction)
    configure_yaml(parser, "emotion_late_fusion", "experiments/pmemo/late_fusion.yaml")
    parser.set_defaults(command_handler=run_from_args)


def config_from_args(args):
    base = mid.config_from_args(args)
    reference = args.reference_output_dir.expanduser().resolve()
    if "output_dir" in args.yaml_provenance["cli_overrides"] and "reference_output_dir" not in args.yaml_provenance["cli_overrides"]:
        reference = base.output_dir.parent / f"{base.output_dir.name}_reference"
    return EmotionLateFusionConfig(**{item.name: getattr(base, item.name) for item in fields(mid.EmotionMidFusionConfig)},
                                  reference_output_dir=reference)


def run_from_args(args):
    try:
        run_late_fusion(config_from_args(args))
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        LOGGER.error("PMEmo late-fusion: %s", error)
        return 2
    return 0


def direct_main(argv=None):
    parser = YAMLArgumentParser(prog="train_late.py")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    configure_parser(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    return run_from_args(args)

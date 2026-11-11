"""Native-score emotion mid fusion through the unchanged PMEmo fitting protocol."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields, replace
from pathlib import Path

from stemfusion.configuration import YAMLArgumentParser, configure_yaml
from stemfusion.data.emotion_mid_fusion import TARGETS
from stemfusion.data.emotion_fusion import label_policy_for, prepare_emotion_features
from stemfusion.data.deam import read_feature_manifest as read_deam_manifest
from stemfusion.experiments import _emotion_recognition as pmemo
from stemfusion.features.emotion_dataset import EmotionDatasetAdapter
from stemfusion.data.jamendo_mid_fusion import JamendoMidFusionDataset
from stemfusion.models.music2emo_emotion import Music2EmoEmotionMidFusion, parameter_count
from stemfusion.training import music2emo
from stemfusion.training.emotion import BinaryEmotionTask
from stemfusion.training.reporting import file_checksum, fingerprint, summarize_results, write_json

LOGGER = logging.getLogger(__name__)
SEEDS = pmemo.SEEDS


@dataclass(frozen=True)
class EmotionMidFusionConfig(pmemo.EmotionMidFusionConfig):
    dataset: str = "deam"


def dataset_adapter(dataset):
    if dataset != "deam":
        raise ValueError("This native-score training entry currently supports DEAM only.")
    return EmotionDatasetAdapter(dataset, read_deam_manifest)


def implementation_identity(dataset="deam"):
    package = Path(__file__).resolve().parents[1]
    adapter = dataset_adapter(dataset)
    names = ("experiments/_emotion_mid_fusion.py", "data/emotion_fusion.py", "data/deam.py",
             "features/emotion_dataset.py")
    return fingerprint({"shared_fitting_dependencies": pmemo.implementation_identity(),
                        "dataset_adapter": adapter.identity(), "label_policy": label_policy_for(dataset),
                        **{name: file_checksum(package / name) for name in names}})


def _plan(config):
    dataset_adapter(config.dataset)
    plan = pmemo._plan(config)
    plan.update(dataset=config.dataset, label_policy=label_policy_for(config.dataset))
    return plan


def format_mid_summary(summary):
    lines = [f"{summary['dataset'].upper()}: binary {summary['target']} | MERT + source-specific chord/key mid-fusion",
             f"Seeds: {', '.join(map(str, summary['seeds']))}; mean +/- sample SD (ddof=1).",
             f"Ground truth: original 1-9 score >= {summary['label_policy']['threshold']:g} is high; lower scores are low. No score normalization.",
             "Checkpoint: minimum validation weighted BCE; threshold: validation macro-F1 grid, ties to lowest.",
             "Accuracy is track-level; macro metrics average low and high classes. PR-AUC means average precision.",
             "Shared chord Transformer; controls repeat the complete encoded mixture vector.", "",
             "Condition | Accuracy (%) | Macro F1 | Macro ROC-AUC | Macro PR-AUC", "--- | --- | --- | --- | ---"]
    def render(value, scale=1):
        sd = value["sample_sd"]
        return f"{value['mean'] * scale:.4f}" + (f" +/- {sd * scale:.4f}" if sd is not None else " (SD unavailable: one seed)")
    def cells(metrics):
        return [render(metrics[key], 100 if key == "accuracy" else 1)
                for key in ("accuracy", "f1_macro", "roc_auc_macro", "pr_auc_macro")]
    for name, metrics in summary["conditions"].items():
        lines.append(" | ".join([name, *cells(metrics)]))
    if summary["paired_differences"]:
        lines.extend(["", "Paired differences: fusion minus its matched repeated-mixture control.",
                      "Comparison | Accuracy (percentage points) | Macro F1 | Macro ROC-AUC | Macro PR-AUC",
                      "--- | --- | --- | --- | ---"])
        for name, comparison in summary["paired_differences"].items():
            lines.append(" | ".join([f"{name} - {comparison['control']}", *cells(comparison["metrics"])]))
    return "\n".join(lines) + "\n"


def run_mid_fusion(config, *, features=None):
    plan = _plan(config)
    label_policy = plan["label_policy"]
    if config.dry_run:
        print(json.dumps({"dry_run": True, **plan}, indent=2, sort_keys=True))
        return {"dry_run": True, **plan}
    if not config.verify_checksums:
        LOGGER.warning("Feature payload checksums are disabled; trusting completion markers. Finite cache changes may go undetected. "
                       "ID/split/shape/finiteness/settings/alignment checks remain mandatory.")
    if features is None:
        features = prepare_emotion_features(config.manifest, config.mert_dir, config.harmony_dir,
            config.chord_root, config.chord_attr, tuple(plan["sources"]), adapter=dataset_adapter(config.dataset), verify_checksums=config.verify_checksums)
    if features.audit.get("dataset") != config.dataset or features.label_policy != label_policy:
        raise ValueError("Feature dataset/label policy differs from the selected native-score experiment.")
    target_data = {target: features.target(target) for target in config.targets}
    if config.validate_only:
        result = {"validate_only": True, "plan": plan, "targets": {target: data.audit for target, data in target_data.items()}}
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    implementation = implementation_identity(config.dataset)
    summaries = {}
    for target, data in target_data.items():
        LOGGER.info("%s %s: %s", config.dataset.upper(), target, json.dumps(data.audit, sort_keys=True))
        results = {name: {} for name in plan["conditions"]}
        task = BinaryEmotionTask(target)
        for seed in config.seeds:
            for name, condition in plan["conditions"].items():
                sources = tuple(condition["sources"])
                count = condition.get("matched_sources", len(sources))
                repeated = condition["kind"] == "repeated_mixture"
                fit_config = replace(config, targets=(target,), output_dir=config.output_dir / target / name)
                result = music2emo.fit_model(fit_config, data.condition(sources), seed,
                    model_factory=lambda: Music2EmoEmotionMidFusion(count, repeated_mixture=repeated),
                    dataset_factory=lambda records, training: JamendoMidFusionDataset(records, sources=sources, training=training),
                    implementation=implementation, task=task,
                    protocol_updates={"condition": name, "source": list(sources), "inputs": count, "kind": condition["kind"],
                        "label_policy": label_policy, "harmony_encoder": "shared", "cohort_fingerprint": data.cohort_fingerprint,
                        "sampling": "shared_epoch_seeded_conditional_uniform_consecutive_segment_mean",
                        "repeated_mixture": "encode_once_repeat_complete_vector" if repeated else None})
                if result["trainable_parameters"] != parameter_count(count):
                    raise ValueError("Actual model parameters do not match the planned control/fusion architecture.")
                results[name][seed] = {**result, "condition": name, "target": target, "cohort_fingerprint": data.cohort_fingerprint}
        comparisons = {name: f"repeated_mixture_{len(value['sources'])}" for name, value in plan["conditions"].items()
                       if value["kind"] == "fusion"}
        paired = {name: {seed: {**result, "input_fingerprint": result["data_fingerprint"], "data_fingerprint": data.cohort_fingerprint}
                         for seed, result in runs.items()} for name, runs in results.items()}
        summary = summarize_results(paired, config.seeds, comparisons)
        summary.update(dataset=config.dataset, target=target, method="mid_fusion", runs=results, plan=plan, data_audit=data.audit,
                       label_policy=label_policy, metric_conventions=task.protocol(), implementation_sha256=implementation,
                       config_provenance=config.config_provenance,
                       parameter_counts={name: parameter_count(value.get("matched_sources", len(value["sources"])))
                                         for name, value in plan["conditions"].items()})
        name = f"summary_mid_fusion_{target}"
        if config.selected_run:
            name += "_run_" + config.selected_run
        if config.seeds != SEEDS:
            name += "_seeds_" + "_".join(map(str, config.seeds))
        path = config.summary_dir / name
        write_json(path.with_suffix(".json"), summary)
        temporary = path.with_name(f".{name}.txt.tmp")
        try:
            temporary.write_text(format_mid_summary(summary), encoding="utf-8")
            temporary.replace(path.with_suffix(".txt"))
        finally:
            temporary.unlink(missing_ok=True)
        print(format_mid_summary(summary), end="")
        LOGGER.info("Saved results: %s.{json,txt}", path)
        summaries[target] = summary
    return {"targets": summaries, "plan": plan}


def configure_parser(parser, *, dataset="deam"):
    dataset_adapter(dataset)
    for name in ("manifest", "mert-dir", "harmony-dir", "chord-root", "chord-attr", "output-dir", "summary-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, action="append", dest="targets", help="Repeat to select targets; default: both.")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    for name in ("epochs", "batch-size", "num-workers", "early-stopping-patience"):
        parser.add_argument(f"--{name}", type=int)
    for name in ("learning-rate", "early-stopping-min-delta"):
        parser.add_argument(f"--{name}", type=float)
    parser.add_argument("--device")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--resume", action="store_true")
    output.add_argument("--overwrite", action="store_true")
    checksums = parser.add_mutually_exclusive_group()
    checksums.add_argument("--verify-checksums", dest="verify_checksums", action="store_true")
    checksums.add_argument("--no-verify-checksums", dest="verify_checksums", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    configure_yaml(parser, f"{dataset}_mid_fusion", f"experiments/{dataset}/mid_fusion.yaml")
    parser.set_defaults(command_handler=run_from_args, dataset=dataset)


def config_from_args(args):
    base = pmemo.config_from_args(args)
    return EmotionMidFusionConfig(**{item.name: getattr(base, item.name) for item in fields(pmemo.EmotionMidFusionConfig)},
                                  dataset=args.dataset)


def run_from_args(args):
    try:
        run_mid_fusion(config_from_args(args))
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        LOGGER.error("%s mid-fusion: %s", args.dataset.upper(), error)
        return 2
    return 0


def direct_main(argv=None, *, dataset="deam"):
    parser = YAMLArgumentParser(prog="train_mid.py")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    configure_parser(parser, dataset=dataset)
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    return run_from_args(args)

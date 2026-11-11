"""Binary-emotion experiment scheduling; fitting stays in training/music2emo.py."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

from stemfusion.configuration import JAMENDO_MID_CONDITIONS as MID_CONDITIONS, YAMLArgumentParser, configure_yaml
from stemfusion.data.emotion_mid_fusion import LABEL_POLICY, TARGETS, prepare_emotion_features
from stemfusion.data.jamendo_mid_fusion import SOURCES, JamendoMidFusionDataset
from stemfusion.models.music2emo_emotion import Music2EmoEmotionMidFusion, parameter_count
from stemfusion.training import music2emo
from stemfusion.training.emotion import BinaryEmotionTask
from stemfusion.training.reporting import file_checksum, fingerprint, summarize_results, write_json

LOGGER = logging.getLogger(__name__)
SEEDS = (32, 64, 128, 256, 512)


@dataclass(frozen=True)
class EmotionMidFusionConfig:
    manifest: Path
    mert_dir: Path
    harmony_dir: Path
    chord_root: Path
    chord_attr: Path
    output_dir: Path
    summary_dir: Path
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
    verify_checksums: bool = True
    conditions: dict = field(default_factory=lambda: json.loads(json.dumps(MID_CONDITIONS)))
    selected_run: str | None = None
    config_provenance: dict | None = None


def implementation_identity():
    package = Path(__file__).resolve().parents[1]
    names = ("experiments/_emotion_recognition.py", "data/emotion_mid_fusion.py", "data/pmemo.py",
             "data/jamendo_mid_fusion.py", "data/jamendo_late_fusion.py", "data/jamendo_reproduction.py", "data/segments.py",
             "models/music2emo.py", "models/music2emo_mid_fusion.py", "models/music2emo_emotion.py",
             "training/music2emo.py", "training/emotion.py", "training/trainer.py", "training/reporting.py")
    return fingerprint({name: file_checksum(package / name) for name in names})


def _plan(config):
    if (not config.targets or len(set(config.targets)) != len(config.targets) or set(config.targets) - set(TARGETS)
            or not config.seeds or len(set(config.seeds)) != len(config.seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in config.seeds)):
        raise ValueError("Choose distinct valid targets and outer seeds.")
    if (type(config.verify_checksums) is not bool or config.epochs <= 0 or config.batch_size <= 0 or config.num_workers < 0
            or not math.isfinite(config.learning_rate) or config.learning_rate <= 0
            or type(config.early_stopping_patience) is not int or config.early_stopping_patience < 0
            or not math.isfinite(config.early_stopping_min_delta) or config.early_stopping_min_delta < 0):
        raise ValueError("Invalid training, stopping or verification settings.")
    if not config.conditions or any(MID_CONDITIONS.get(name) != value for name, value in config.conditions.items()):
        raise ValueError("Use the shared fixed seven-condition mid-fusion matrix.")
    if config.selected_run is not None and config.selected_run not in config.conditions:
        raise ValueError(f"Unknown condition: {config.selected_run}")
    conditions = {config.selected_run: config.conditions[config.selected_run]} if config.selected_run else config.conditions
    sources = tuple(source for source in SOURCES if any(source in row["sources"] for row in conditions.values()))
    return {"dataset": "pmemo", "targets": list(config.targets), "conditions": conditions, "sources": list(sources),
            "seeds": list(config.seeds), "label_policy": LABEL_POLICY, "epochs": config.epochs,
            "batch_size": config.batch_size, "learning_rate": config.learning_rate,
            "early_stopping": {"monitor": "val_loss", "patience": config.early_stopping_patience,
                               "min_delta": config.early_stopping_min_delta},
            "checkpoint": "minimum_validation_weighted_bce", "verify_checksums": config.verify_checksums,
            "output_dir": str(config.output_dir), "summary_dir": str(config.summary_dir)}


def format_mid_summary(summary):
    lines = [f"PMEmo: binary {summary['target']} | MERT + source-specific chord/key mid-fusion",
             f"Seeds: {', '.join(map(str, summary['seeds']))}; mean +/- sample SD (ddof=1).",
             "Ground truth: original 0-1 score >= 0.5 is high; score < 0.5 is low.",
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
        LOGGER.info("PMEmo %s: %s", target, json.dumps(data.audit, sort_keys=True))
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
                        "label_policy": LABEL_POLICY, "harmony_encoder": "shared", "cohort_fingerprint": data.cohort_fingerprint,
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
        summary.update(dataset="pmemo", target=target, method="mid_fusion", runs=results, plan=plan, data_audit=data.audit,
                       label_policy=LABEL_POLICY, metric_conventions=task.protocol(), implementation_sha256=implementation,
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


def configure_parser(parser):
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
    configure_yaml(parser, "emotion_mid_fusion", "experiments/pmemo/mid_fusion.yaml")
    parser.set_defaults(command_handler=run_from_args)


def config_from_args(args):
    paths = {name: getattr(args, name).expanduser().resolve() for name in (
        "manifest", "mert_dir", "harmony_dir", "chord_root", "chord_attr", "output_dir", "summary_dir")}
    if "output_dir" in args.yaml_provenance["cli_overrides"] and "summary_dir" not in args.yaml_provenance["cli_overrides"]:
        paths["summary_dir"] = paths["output_dir"]
    return EmotionMidFusionConfig(**paths, targets=tuple(args.targets), seeds=tuple(args.seeds),
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience, early_stopping_min_delta=args.early_stopping_min_delta,
        device=args.device, num_workers=args.num_workers, resume=args.resume and not args.overwrite,
        overwrite=args.overwrite, dry_run=args.dry_run, validate_only=args.validate_only,
        verify_checksums=args.verify_checksums, conditions=args.runs, selected_run=args.run, config_provenance=args.yaml_provenance)


def run_from_args(args):
    try:
        run_mid_fusion(config_from_args(args))
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        LOGGER.error("PMEmo mid-fusion: %s", error)
        return 2
    return 0


def direct_main(argv=None):
    parser = YAMLArgumentParser(prog="train_mid.py")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    configure_parser(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    return run_from_args(args)

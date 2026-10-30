"""FMA-Medium experiment configuration, scheduling, and dataset handoff."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from stemfusion.configuration import YAMLArgumentParser, configure_yaml
from stemfusion.data.fma_features import SOURCES, cache_paths, inspect_manifest, load_fma_features
from stemfusion.training.late_fusion import assemble_ensemble, branch_seed, train_branch
from stemfusion.training.metrics import GenreMetrics
from stemfusion.training.mid_fusion import match_capacity, model_parameter_count, train_mid_condition
from stemfusion.training.reporting import fingerprint, summarize_results, write_json
from stemfusion.training.trainer import TrainingSettings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FMAExperimentConfig:
    manifest_path: Path
    feature_directory: Path
    output_directory: Path
    family: str
    conditions: dict[str, dict]
    seeds: tuple[int, ...]
    settings: TrainingSettings = field(default_factory=TrainingSettings)
    dimension: int = 1024
    number_of_classes: int = 16
    selected_run: str | None = None
    dry_run: bool = False
    validate_only: bool = False
    resume: bool = True
    overwrite: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)
    summary_directory: Path | None = None


def configure_parser(parser: argparse.ArgumentParser, family: str) -> None:
    parser.description = f"Run the FMA-Medium {family.replace('_', '-')} matrix with matched controls and fixed outer seeds."
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Common experiment root; branches can be shared between mid and late fusion.")
    parser.add_argument("--summary-dir", type=Path, help="Destination for readable and detailed summaries. Defaults to the YAML path; an explicit --output-dir also redirects summaries unless this option is supplied.")
    parser.add_argument("--seed", type=int, action="append", dest="seeds", help="Repeat for a seed subset; replaces the YAML seed list.")
    parser.add_argument("--expected-feature-dimension", type=int)
    parser.add_argument("--hidden-dimensions", type=int, nargs=2)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Show the plan and check local paths without reading feature contents or training.")
    parser.add_argument("--validate-only", action="store_true", help="Check all feature contents and reports without training or writing outputs.")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--resume", action="store_true", default=True,
                             help="Reuse verified complete runs (the default); accepted for compatibility.")
    output_mode.add_argument("--overwrite", action="store_true", help="Retrain and replace existing outputs instead of reusing completed runs.")
    configure_yaml(parser, family, f"experiments/fma_medium/{family}.yaml")
    parser.set_defaults(fusion_family=family, command_handler=run_from_args)


def config_from_args(args: argparse.Namespace) -> FMAExperimentConfig:
    summary_directory = args.summary_dir
    explicit = args.yaml_provenance["cli_overrides"]
    if "output_dir" in explicit and "summary_dir" not in explicit:
        summary_directory = args.output_dir
    return FMAExperimentConfig(
        manifest_path=args.manifest.expanduser().resolve(), feature_directory=args.feature_dir.expanduser().resolve(),
        output_directory=args.output_dir.expanduser().resolve(), family=args.fusion_family,
        conditions=args.yaml_provenance["experiment_metadata"].get("runs", {}), seeds=tuple(args.seeds),
        settings=TrainingSettings(hidden_dimensions=tuple(args.hidden_dimensions), dropout=args.dropout,
            learning_rate=args.learning_rate, batch_size=args.batch_size, epochs=args.epochs, device=args.device, num_workers=args.num_workers),
        dimension=args.expected_feature_dimension, selected_run=args.run, dry_run=args.dry_run,
        validate_only=args.validate_only, resume=args.resume and not args.overwrite,
        overwrite=args.overwrite, provenance=args.yaml_provenance,
        summary_directory=summary_directory.expanduser().resolve() if summary_directory is not None else None,
    )


def selected_conditions(config: FMAExperimentConfig) -> dict:
    config.settings.validate()
    if config.family not in {"mid_fusion", "late_fusion"}:
        raise ValueError("Unknown fusion family.")
    if not config.seeds or len(set(config.seeds)) != len(config.seeds) or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in config.seeds):
        raise ValueError("Outer seeds must be distinct integers in [0, 2**32).")
    if config.dimension <= 0:
        raise ValueError("Feature dimension must be positive.")
    if not config.conditions:
        raise ValueError("Use an experiment YAML containing named runs.")
    if config.selected_run is not None and config.selected_run not in config.conditions:
        raise ValueError(f"Unknown condition: {config.selected_run}")
    selected = config.conditions if config.selected_run is None else {config.selected_run: config.conditions[config.selected_run]}
    for name, condition in selected.items():
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Condition names must be safe directory names.")
        sources = tuple(condition["sources"])
        kind = condition.get("kind")
        count = condition.get("matched_sources")
        if kind == "reference" and config.family == "mid_fusion" and sources == ("mixture",) and count is None:
            continue
        if kind == "fusion" and sources in (SOURCES[:4], SOURCES[:5], SOURCES) and count is None:
            continue
        expected_kind = "capacity_matched" if config.family == "mid_fusion" else "mixture_ensemble"
        if kind == expected_kind and sources == ("mixture",) and type(count) is int and count in (4, 5, 6):
            continue
        raise ValueError(f"Condition {name!r} does not match the agreed FMA protocol.")
    for seed in config.seeds:
        values = [branch_seed(seed, index) for index in range(6)]
        if len(set(values)) != len(values):
            raise ValueError("Derived branch seeds collide.")
    return selected


def plan_experiment(config: FMAExperimentConfig) -> dict:
    conditions = selected_conditions(config)
    records, mapping = inspect_manifest(config.manifest_path, config.number_of_classes)
    paths = cache_paths(config.feature_directory)
    planned = {}
    for name, condition in conditions.items():
        count = len(condition["sources"])
        detail = dict(condition)
        if condition["kind"] == "capacity_matched":
            detail["capacity_matching"] = match_capacity(config.dimension, len(mapping), condition["matched_sources"], config.settings.hidden_dimensions)
            detail["trainable_parameters"] = detail["capacity_matching"]["matched_parameters"]
        elif config.family == "mid_fusion":
            detail["trainable_parameters"] = model_parameter_count(config.dimension * count, len(mapping), config.settings.hidden_dimensions)
        else:
            count = condition.get("matched_sources", count)
            detail["trainable_parameters"] = count * model_parameter_count(config.dimension, len(mapping), config.settings.hidden_dimensions)
        planned[name] = detail
    return {"family": config.family, "seeds": list(config.seeds), "conditions": planned,
            "manifest": str(config.manifest_path), "manifest_split_counts": {split: sum(record.split == split for record in records) for split in ("train", "validation", "test")},
            "feature_caches": {source: {split: str(path) for split, path in splits.items()} for source, splits in paths.items()},
            "output_directory": str(config.output_directory), "summary_directory": str(config.summary_directory or config.output_directory), "settings": vars(config.settings),
            "branch_seed_rule": "index 0 = outer seed; otherwise first 32 SHA256 bits of stemfusion-phase2:<outer>:<index>",
            "branch_seeds": {str(seed): [branch_seed(seed, index) for index in range(6)] for seed in config.seeds},
            "aggregation": "sum_logits_then_softmax", "dry_run": config.dry_run}


def run_experiment(config: FMAExperimentConfig, *, data=None) -> dict:
    if config.overwrite:
        config = replace(config, resume=False)
    plan = plan_experiment(config)
    if config.dry_run:
        return plan
    conditions = selected_conditions(config)
    if data is None:
        data = load_fma_features(config.manifest_path, config.feature_directory, config.dimension, config.number_of_classes)
    if data.dimension != config.dimension or data.number_of_classes != config.number_of_classes:
        raise ValueError("Prepared features must match the configured dimension and class count.")
    if config.validate_only:
        return {"validation_only": True, "data_fingerprint": data.fingerprint, **data.audit}
    task = GenreMetrics(data.number_of_classes)
    write_json(config.output_directory / f"data_audit_{data.fingerprint}.json", data.audit)
    write_json(config.output_directory / "plans" / f"{config.family}_{fingerprint(plan)}.json", {**plan, "config_provenance": config.provenance})
    results = {name: {} for name in conditions}
    baseline_results = {}
    for seed in config.seeds:
        branches = {}

        def branch(source: str, index: int) -> dict:
            key = (source, index)
            if key not in branches:
                branches[key] = train_branch(data=data, source=source, index=index, outer_seed=seed,
                    root=config.output_directory, settings=config.settings, task=task, provenance=config.provenance,
                    resume=config.resume, overwrite=config.overwrite)
            return branches[key]

        for name, condition in conditions.items():
            LOGGER.info("%s | outer seed %d | %s", config.family, seed, name)
            directory = config.output_directory / config.family / name / f"seed_{seed}"
            if condition["kind"] == "reference":
                result = branch("mixture", 0)
            elif config.family == "mid_fusion":
                result = train_mid_condition(data=data, sources=tuple(condition["sources"]), settings=config.settings,
                    seed=seed, output_directory=directory, provenance=config.provenance, task=task,
                    matched_sources=condition.get("matched_sources"), resume=config.resume, overwrite=config.overwrite)
            else:
                if condition["kind"] == "mixture_ensemble":
                    members = [branch("mixture", index) for index in range(condition["matched_sources"])]
                else:
                    members = [branch(source, index) for index, source in enumerate(condition["sources"])]
                result = assemble_ensemble(branches=members, output_directory=directory, task=task,
                    outer_seed=seed, condition=name, resume=config.resume, overwrite=config.overwrite)
                baseline_results[seed] = branch("mixture", 0)
            results[name][seed] = {**result, "outer_seed": seed}
    comparisons = {}
    for name, condition in conditions.items():
        if condition["kind"] != "fusion":
            continue
        target_kind = "capacity_matched" if config.family == "mid_fusion" else "mixture_ensemble"
        for control, candidate in conditions.items():
            if candidate["kind"] == target_kind and candidate["matched_sources"] == len(condition["sources"]):
                comparisons[name] = control
    summary = summarize_results(results, config.seeds, comparisons)
    if baseline_results:
        summary["mixture_only_reference"] = summarize_results({"mixture_only": baseline_results}, config.seeds, {})["conditions"]["mixture_only"]
    summary.update({"family": config.family, "data_fingerprint": data.fingerprint, "runs": results,
                    "dataset": "FMA-Medium", "number_of_classes": data.number_of_classes,
                    "feature_dimension": data.dimension, "split_counts": data.audit["split_counts"],
                    "training_settings": vars(config.settings),
                    "historical_selection_note": "The fixed nested stem sets were influenced by historical FMA test results; these reruns are not an independent confirmatory holdout."})
    summary_path = summary_file(config)
    write_json(summary_path, summary)
    write_readable_summary(summary_path, summary)
    LOGGER.info("Saved results: %s", summary_path)
    return summary


def summary_file(config: FMAExperimentConfig) -> Path:
    suffix = "" if config.selected_run is None else f"_{config.selected_run}"
    if config.selected_run is not None or config.seeds != (32, 64, 128, 256, 512):
        suffix += f"_seeds_{'_'.join(map(str, config.seeds))}"
    return (config.summary_directory or config.output_directory) / f"summary_{config.family}{suffix}.json"


def write_readable_summary(json_path: Path, summary: dict) -> Path:
    """Save a publishable plain-text companion without modifying the JSON."""
    path = json_path.with_suffix(".txt")
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_text(format_results(summary) + f"\n\nDetailed JSON: {json_path.name}\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def format_results(summary: dict) -> str:
    """Present FMA results without changing saved metrics or training logic."""
    columns = (("accuracy", "Accuracy"), ("f1_macro", "Macro F1"),
               ("roc_auc_macro", "Macro ROC-AUC"), ("pr_auc_macro", "Macro AP (PR-AUC)"))
    names = {
        "mixture_only": "Mixture only",
        "matched_mixture_4": "Matched mixture (4)",
        "matched_mixture_5": "Matched mixture (5)",
        "matched_mixture_6": "Matched mixture (6)",
        "mixture_ensemble_4": "Mixture ensemble (4)",
        "mixture_ensemble_5": "Mixture ensemble (5)",
        "mixture_ensemble_6": "Mixture ensemble (6)",
        "mixture_guitar_drums_piano": "M+G+D+P",
        "mixture_guitar_drums_piano_bass": "M+G+D+P+B",
        "mixture_all_five_stems": "M+G+D+P+B+V",
    }

    def cells(metrics: dict, signed: bool = False) -> list[str]:
        values = []
        for key, _ in columns:
            metric = metrics[key]
            mean = format(metric["mean"], "+.4f" if signed else ".4f")
            sd = metric["sample_sd"]
            values.append(mean if sd is None else f"{mean} +/- {sd:.4f}")
        return values

    def table(label: str, rows: list[list[str]]) -> str:
        header = [label, *(title for _, title in columns)]
        widths = [max(len(row[index]) for row in [header, *rows]) for index in range(len(header))]
        formatted = [" | ".join(value.ljust(width) for value, width in zip(header, widths)),
                     "-+-".join("-" * width for width in widths)]
        formatted.extend(" | ".join(value.ljust(width) for value, width in zip(row, widths)) for row in rows)
        return "\n".join(formatted)

    family = summary["family"].replace("_", " ").capitalize()
    lines = [f"{family} test results", "Dataset: FMA-Medium | Task: 16-class genre classification",
             "Features: frozen MusicFM-MSD, layer 7; HT-Demucs htdemucs_6s stems.",
             "Outer seeds: " + ", ".join(map(str, summary["seeds"]))]
    if "number_of_classes" in summary:
        lines[1] = f"Dataset: {summary['dataset']} | Task: {summary['number_of_classes']}-class genre classification"
    if "feature_dimension" in summary:
        lines.append(f"Embedding width per source: {summary['feature_dimension']}")
    if "split_counts" in summary:
        counts = summary["split_counts"]
        lines.append(f"Tracks: train {counts['train']}, validation {counts['validation']}, test {counts['test']}.")
    if "training_settings" in summary:
        settings = summary["training_settings"]
        hidden = "/".join(map(str, settings["hidden_dimensions"]))
        lines.extend([f"Training: Adam, learning rate {settings['learning_rate']:g}, batch size {settings['batch_size']}, {settings['epochs']} epochs.",
                      f"Standard MLP hidden widths: {hidden}; ReLU; dropout {settings['dropout']:g}.",
                      "Checkpoint selection: maximum validation accuracy + macro ROC-AUC + macro AP; earliest epoch on ties."])
    if len(summary["seeds"]) > 1:
        lines.append("Mean +/- sample SD (ddof=1); metrics on the 0-1 scale.")
    else:
        lines.append("Single-seed metrics on the 0-1 scale; sample SD is unavailable.")
    metrics = dict(summary["conditions"])
    if "mixture_only_reference" in summary and "mixture_only" not in metrics:
        metrics = {"mixture_only": summary["mixture_only_reference"], **metrics}
    lines.extend(["", table("Condition", [[names.get(name, name), *cells(values)] for name, values in metrics.items()])])
    paired = summary["paired_differences"]
    if paired:
        rows = [[f"{names.get(name, name)} vs {names.get(values['control'], values['control'])}",
                 *cells(values["metrics"], signed=True)] for name, values in paired.items()]
        lines.extend(["", "Paired differences (fusion minus matched control):", table("Comparison", rows)])
    lines.extend(["", "Reading the results:",
                  "M = original mixture; G = guitar; D = drums; P = piano; B = bass; V = vocals.",
                  "The native 'other' stem is excluded. M+G+D+P+B+V uses all five selected stems.",
                  "Matched mixture (k): one mixture-only MLP with approximately the parameter count of k-source mid-fusion.",
                  "Mixture ensemble (k): k independently trained mixture-only classifiers.",
                  "Late fusion sums selected branch logits equally, then applies softmax.",
                  "PR-AUC here means macro average precision; F1 and ROC-AUC are also macro averaged.",
                  "Paired differences use the same outer seed; positive values favor stem fusion.",
                  "Sample SD describes training-run variability. No significance tests were performed."])
    return "\n".join(lines)


def run_from_args(args: argparse.Namespace) -> int:
    try:
        config = config_from_args(args)
        result = run_experiment(config)
        if config.dry_run:
            print(json.dumps(result, indent=2, sort_keys=True))
        elif config.validate_only:
            print(json.dumps({key: result[key] for key in ("validation_only", "data_fingerprint", "split_counts", "dimension", "number_of_classes")}, indent=2))
        else:
            print(format_results(result))
            print(f"\nReadable report: {summary_file(config).with_suffix('.txt')}")
            print(f"\nDetailed results: {summary_file(config)}")
    except (ValueError, OSError) as error:
        LOGGER.error("%s", error)
        return 2
    return 0


def direct_main(family: str, argv=None) -> int:
    parser = YAMLArgumentParser(prog="train_mid.py" if family == "mid_fusion" else "train_late.py")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    configure_parser(parser, family)
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    return run_from_args(args)

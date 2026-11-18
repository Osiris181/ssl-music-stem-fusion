"""Train GTZAN CNN-feature mid fusion using the shared genre fit backend."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import yaml

from stemfusion.configuration import load_configuration, _UniqueKeyLoader
from stemfusion.experiments._genre import (
    GenreExperimentConfig, run_experiment, format_results, summary_file,
)
from stemfusion.training.reporting import file_checksum
from stemfusion.training.trainer import TrainingSettings

DEFAULT_CONFIG = Path(__file__).resolve().parents[4] / "configs/experiments/gtzan/mid_fusion_cnn.yaml"


def config_from_args(args):
    path = args.config.resolve()
    raw = yaml.load(path.read_text(), Loader=_UniqueKeyLoader)
    if not isinstance(raw, dict) or set(raw) != {"protocol", "dataset", "features", "paths"} or \
       raw["dataset"] != "gtzan" or raw["features"] != {"name": "msd_musicnn", "dimension_per_source": 200}:
        raise ValueError("Expected a GTZAN MSD_musicnn configuration with 200 features per source")
    defaults, provenance = load_configuration((path.parent / raw["protocol"]).resolve(), "mid_fusion")
    if provenance["experiment_metadata"]["dataset"] != "gtzan":
        raise ValueError("CNN protocol must inherit the GTZAN genre experiment")
    if set(raw["paths"]) != {"manifest", "feature_dir", "output_dir", "summary_dir"}:
        raise ValueError("Missing or unsupported CNN training paths")
    paths = {key: (path.parent / value).resolve() for key, value in raw["paths"].items()}
    overrides = {}
    for key in paths:
        if getattr(args, key) is not None:
            paths[key] = getattr(args, key).resolve()
            overrides[key] = str(paths[key])
    if args.output_dir is not None and args.summary_dir is None:
        paths["summary_dir"] = paths["output_dir"]
    for key in ("seeds", "hidden_dimensions", "dropout", "learning_rate", "batch_size", "epochs", "device", "num_workers"):
        value = getattr(args, key)
        if value is not None:
            defaults[key] = value
            overrides[key] = value
    provenance.update({"cnn_config": {"path": str(path), "sha256": file_checksum(path)},
                       "feature_backend": "msd_musicnn", "resolved_cnn_paths": {k: str(v) for k, v in paths.items()},
                       "cli_overrides": overrides,
                       "representation_override": {"name": "msd_musicnn", "dimension_per_source": 200}})
    settings = TrainingSettings(**{key: tuple(defaults[key]) if key == "hidden_dimensions" else defaults[key]
                                  for key in ("hidden_dimensions", "dropout", "learning_rate", "batch_size", "epochs", "device", "num_workers")})
    return GenreExperimentConfig(
        manifest_path=paths["manifest"], feature_directory=paths["feature_dir"],
        output_directory=paths["output_dir"], summary_directory=paths["summary_dir"],
        family="mid_fusion", dataset="gtzan", feature_backend="msd_musicnn", dimension=200,
        number_of_classes=10, conditions=provenance["experiment_metadata"]["runs"],
        seeds=tuple(defaults["seeds"]), settings=settings, selected_run=args.run,
        dry_run=args.dry_run, validate_only=args.validate_only,
        resume=not args.overwrite, overwrite=args.overwrite, provenance=provenance,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for name in ("manifest", "feature-dir", "output-dir", "summary-dir"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--run")
    parser.add_argument("--hidden-dimensions", type=int, nargs=2)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--learning-rate", type=float)
    for name in ("batch-size", "epochs", "num-workers"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Reuse verified complete runs (the default)")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        config = config_from_args(args)
        result = run_experiment(config)
        if config.dry_run or config.validate_only:
            if config.validate_only:
                result = {key: result[key] for key in
                          ("validation_only", "data_fingerprint", "split_counts", "dimension", "number_of_classes")}
            print(json.dumps(result, indent=2, default=str))
        else:
            print(format_results(result))
            print(f"\nReadable report: {summary_file(config).with_suffix('.txt')}\nDetailed results: {summary_file(config)}")
    except (ValueError, OSError) as error:
        logging.error("%s", error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run only the five-seed, single-dataset Music2Emo mixture baseline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.configuration import YAMLArgumentParser, configure_yaml


def configure_parser(parser):
    for name in ("manifest", "mert-dir", "harmony-dir", "tag-list", "author-metadata", "output-dir", "summary-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", dest="seeds", type=int, action="append")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--early-stopping-min-delta", type=float)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    configure_yaml(parser, "jamendo_reproduction", "experiments/mtg_jamendo/reproduction.yaml")
    parser.set_defaults(command_handler=run_from_args)


def run_from_args(args):
    from stemfusion.experiments.mtg_jamendo.reproduction import ReproductionConfig, run_reproduction

    try:
        paths = {name: getattr(args, name).expanduser().resolve() for name in (
            "manifest", "mert_dir", "harmony_dir", "tag_list", "author_metadata", "output_dir", "summary_dir")}
        config = ReproductionConfig(**paths, seeds=tuple(args.seeds), epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, device=args.device, num_workers=args.num_workers,
            early_stopping_patience=args.early_stopping_patience, early_stopping_min_delta=args.early_stopping_min_delta,
            resume=args.resume, overwrite=args.overwrite, dry_run=args.dry_run,
            validate_only=args.validate_only, config_provenance=args.yaml_provenance)
        run_reproduction(config)
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        logging.error("Jamendo author reproduction: %s", error)
        return 2
    return 0


def main(argv=None):
    parser = YAMLArgumentParser(prog="train_reproduction.py")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    configure_parser(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    return run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())

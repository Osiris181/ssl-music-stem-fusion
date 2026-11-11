"""Extract shared BTC chord/key descriptors for the MERT experiments."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.configuration import YAMLArgumentParser, configure_yaml


def configure_parser(parser):
    for name in ("manifest", "output-dir", "author-root", "btc-config", "btc-checkpoint", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source", action="append", choices=("mixture", "guitar", "drums", "piano", "bass", "vocals"))
    parser.add_argument("--split", dest="splits", action="append", choices=("train", "validation", "test"))
    parser.add_argument("--device")
    parser.add_argument("--midi-policy", choices=("music2emotion_released", "complete_intervals"))
    parser.add_argument("--keep-midi", action=argparse.BooleanOptionalAction)
    parser.add_argument("--retry-failures", action=argparse.BooleanOptionalAction,
                        help="Retry supported CQT-tail failures once after the normal pass (default: enabled).")
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    configure_yaml(parser, "chord_key_extraction", "experiments/mtg_jamendo/chords_keys.yaml")
    parser.set_defaults(command_handler=run_from_args)


def run_from_args(args):
    from stemfusion.features.chords_keys import ChordKeyExtractionConfig, extract_chords_keys

    try:
        paths = {name: getattr(args, name).expanduser().resolve() for name in (
            "manifest", "output_dir", "author_root", "btc_config", "btc_checkpoint", "report")}
        config = ChordKeyExtractionConfig(**paths, sources=tuple(args.source), splits=tuple(args.splits),
            device=args.device, midi_policy=args.midi_policy, keep_midi=args.keep_midi,
            limit_per_split=args.limit_per_split, overwrite=args.overwrite, fail_fast=args.fail_fast,
            dry_run=args.dry_run, config_provenance=args.yaml_provenance, retry_failures=args.retry_failures)
        report = extract_chords_keys(config)
    except (ValueError, OSError, ImportError, KeyError, TypeError) as error:
        logging.error("Chord/key extraction: %s", error)
        return 2
    logging.info("BTC chords/key: planned=%d, extracted=%d, reused=%d, repaired=%d, failed=%d, unavailable keys=%d%s",
                 report["planned"], report["extracted"], report["reused"], report["repaired"], report["failed"],
                 report["key_unavailable"], " (dry run)" if report["dry_run"] else "")
    return report["exit_code"]


def main(argv=None):
    parser = YAMLArgumentParser(prog="extract_chords_keys.py")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    configure_parser(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    return run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())

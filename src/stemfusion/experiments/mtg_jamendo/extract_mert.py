"""Extract frozen MERT layers separately from whole-track chord/key descriptors."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.configuration import YAMLArgumentParser, configure_yaml
from stemfusion.data.fma import HTDEMUCS_SOURCES


def configure_parser(parser):
    for name in ("manifest", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("model-path", "report"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--source", choices=("mixture", *HTDEMUCS_SOURCES), action="append")
    parser.add_argument("--split", dest="splits", choices=("train", "validation", "test"), action="append")
    parser.add_argument("--device")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction)
    parser.add_argument("--segment-seconds", type=float)
    parser.add_argument("--stem-alignment", choices=("mixture_duration", "none"))
    parser.add_argument("--max-alignment-seconds", type=float)
    parser.add_argument("--short-segment-policy", choices=("pad_to_minimum", "error"))
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    configure_yaml(parser, "mert_segment_extraction", "experiments/mtg_jamendo/mert.yaml")
    parser.set_defaults(command_handler=run_from_args)


def run_from_args(args):
    from stemfusion.features.mert_segments import MERTSegmentConfig, extract_mert_segments

    try:
        config = MERTSegmentConfig(
            manifest_path=args.manifest.expanduser().resolve(), output_dir=args.output_dir.expanduser().resolve(),
            model_path=args.model_path.expanduser().resolve() if args.model_path else None,
            model_name=args.model_name, revision=args.revision, local_files_only=args.local_files_only,
            device=args.device, sample_rate=args.sample_rate, segment_seconds=args.segment_seconds,
            minimum_input_samples=args.minimum_input_samples, short_segment_policy=args.short_segment_policy,
            stem_alignment=args.stem_alignment, max_alignment_seconds=args.max_alignment_seconds,
            splits=tuple(args.splits), limit_per_split=args.limit_per_split, overwrite=args.overwrite,
            continue_on_error=not args.fail_fast, dry_run=args.dry_run,
            report_path=args.report.expanduser().resolve() if args.report else None,
            config_provenance=args.yaml_provenance)
        result = extract_mert_segments(config, tuple(args.source))
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        logging.error("Jamendo MERT extraction: %s", error)
        return 2
    logging.info("MERT segments: planned=%d, extracted=%d, reused=%d, failed=%d, unprocessed=%d, count mismatches=%d%s",
                 result["planned"], result["extracted"], result["reused"], result["failed"], result["unprocessed"],
                 len(result["segment_count_mismatches"]), " (dry run)" if result["dry_run"] else "")
    return result["exit_code"]


def main(argv=None):
    parser = YAMLArgumentParser(prog="extract_mert.py")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    configure_parser(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    return run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())

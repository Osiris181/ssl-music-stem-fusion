"""PMEmo preset routing; model and signal-processing code remains shared."""

from __future__ import annotations

import logging

from stemfusion.configuration import YAMLArgumentParser
from stemfusion.experiments.mtg_jamendo import extract_mert as jamendo_mert
from stemfusion.experiments.mtg_jamendo import extract_chords_keys as jamendo_harmony


def configure_parser(parser, method):
    module = jamendo_mert if method == "mert" else jamendo_harmony
    module.configure_parser(parser)
    parser.yaml_default = f"experiments/pmemo/{method}.yaml"
    for action in parser._actions:
        if action.dest == "config":
            action.help = f"YAML preset. Default: configs/{parser.yaml_default}"
    parser.set_defaults(command_handler=run_from_args, pmemo_method=method)


def run_from_args(args):
    from stemfusion.features.emotion_extraction import extract_chords_keys, extract_mert_segments
    from stemfusion.features.mert_segments import MERTSegmentConfig
    from stemfusion.features.chords_keys import ChordKeyExtractionConfig

    try:
        if args.yaml_provenance["yaml_settings_before_cli_overrides"]["dataset"] != "pmemo":
            raise ValueError("PMEmo entry points require a PMEmo YAML preset.")
        if args.pmemo_method == "mert":
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
            report = extract_mert_segments(config, tuple(args.source))
        else:
            paths = {name: getattr(args, name).expanduser().resolve() for name in (
                "manifest", "output_dir", "author_root", "btc_config", "btc_checkpoint", "report")}
            config = ChordKeyExtractionConfig(**paths, sources=tuple(args.source), splits=tuple(args.splits),
                device=args.device, midi_policy=args.midi_policy, keep_midi=args.keep_midi,
                limit_per_split=args.limit_per_split, overwrite=args.overwrite, fail_fast=args.fail_fast,
                dry_run=args.dry_run, config_provenance=args.yaml_provenance, retry_failures=args.retry_failures)
            report = extract_chords_keys(config)
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        logging.error("PMEmo %s extraction: %s", args.pmemo_method, error)
        return 2
    logging.info("PMEmo %s: planned=%d, extracted=%d, reused=%d, failed=%d, unprocessed=%d%s; report=%s",
                 args.pmemo_method, report["planned"], report["extracted"], report["reused"],
                 report["failed"], report["unprocessed"], " (dry run)" if report["dry_run"] else "",
                 args.report)
    return report["exit_code"]


def main(method, argv=None):
    parser = YAMLArgumentParser(prog=f"extract_{method}.py")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    configure_parser(parser, method)
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    return run_from_args(args)

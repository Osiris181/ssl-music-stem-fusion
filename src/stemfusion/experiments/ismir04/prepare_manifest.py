"""Audit duplicates and preserve the provided ISMIR04 partition assignments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.configuration import YAMLArgumentParser, configure_yaml
from stemfusion.data.ismir04 import prepare_ismir04_manifest


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Audit the CSV-listed audio and generate a fixed-partition ISMIR04 manifest; no random splitting."
    for name in ("metadata-root", "audio-root", "htdemucs-root", "repair-root", "output", "audit-path"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--audit-workers", type=int)
    parser.add_argument("--verified-legacy-stem", action="append", help="Full relative audio path whose colliding legacy stems have been independently verified.")
    parser.add_argument("--audit-only", action="store_true", help="Audit/select tracks and report exclusions without writing a manifest. Reads listed audio; reuses unchanged audit entries.")
    parser.add_argument("--overwrite", action="store_true")
    configure_yaml(parser, "ismir04_manifest", "experiments/ismir04/manifest.yaml")
    parser.set_defaults(command_handler=run_from_args)


def run_from_args(args: argparse.Namespace) -> int:
    try:
        paths = {name: getattr(args, name).expanduser().resolve()
                 for name in ("metadata_root", "audio_root", "htdemucs_root", "repair_root", "output", "audit_path")}
        report = prepare_ismir04_manifest(**paths, audit_workers=args.audit_workers,
            overwrite=args.overwrite, audit_only=args.audit_only,
            verified_legacy_stems=tuple(args.verified_legacy_stem or ()),
            filename_overrides=args.filename_overrides)
        print(json.dumps(report, indent=2, sort_keys=True))
    except (ValueError, OSError) as error:
        print(f"ISMIR04 preparation error: {error}")
        return 2
    return 0


def main(argv=None) -> int:
    parser = YAMLArgumentParser(prog="prepare_manifest.py")
    configure_parser(parser)
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

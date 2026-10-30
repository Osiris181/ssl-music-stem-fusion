"""Generate GTZAN's fixed-split manifest from the provided text files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.configuration import YAMLArgumentParser, configure_yaml
from stemfusion.data.gtzan import GTZANManifestConfig, build_gtzan_manifest


def configure_parser(parser: argparse.ArgumentParser) -> None:
    for name in ("train-split", "validation-split", "test-split", "mixture-root", "htdemucs-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--mixture-extension")
    parser.add_argument("--stem-extension")
    parser.add_argument("--exclude-track-id", action="append", help="Replace the YAML exclusion list; repeat for multiple unavailable tracks.")
    parser.add_argument("--check-audio", action=argparse.BooleanOptionalAction)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    configure_yaml(parser, "gtzan_manifest", "experiments/gtzan/manifest.yaml")
    parser.set_defaults(command_handler=run_from_args)


def run_from_args(args: argparse.Namespace) -> int:
    try:
        config = GTZANManifestConfig(
            split_files={split: getattr(args, f"{split}_split").expanduser().resolve()
                         for split in ("train", "validation", "test")},
            mixture_root=args.mixture_root.expanduser().resolve(),
            htdemucs_root=args.htdemucs_root.expanduser().resolve(),
            output_path=args.output.expanduser().resolve(),
            excluded_track_ids=tuple(args.exclude_track_id or ()),
            mixture_extension=args.mixture_extension, stem_extension=args.stem_extension,
            check_audio=args.check_audio, overwrite=args.overwrite, dry_run=args.dry_run,
        )
        print(json.dumps(build_gtzan_manifest(config), indent=2, sort_keys=True))
    except (ValueError, OSError) as error:
        print(f"GTZAN manifest error: {error}")
        return 2
    return 0


def main(argv=None) -> int:
    parser = YAMLArgumentParser(prog="prepare_manifest.py")
    configure_parser(parser)
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

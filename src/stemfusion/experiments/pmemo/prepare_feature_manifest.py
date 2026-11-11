"""Bind the supplied PMEmo splits to chorus audio and separated stems."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.configuration import YAMLArgumentParser, configure_yaml
from stemfusion.data.pmemo import build_feature_manifest


def configure_parser(parser):
    for name in ("train-split", "validation-split", "test-split", "annotations", "metadata",
                 "mixture-root", "htdemucs-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    configure_yaml(parser, "pmemo_feature_manifest", "experiments/pmemo/feature_manifest.yaml")
    parser.set_defaults(command_handler=run_from_args)


def run_from_args(args):
    try:
        report = build_feature_manifest(
            split_files={split: getattr(args, f"{split}_split").expanduser().resolve()
                         for split in ("train", "validation", "test")},
            **{name: getattr(args, name).expanduser().resolve() for name in (
                "annotations", "metadata", "mixture_root", "htdemucs_root", "output")},
            overwrite=args.overwrite, dry_run=args.dry_run)
        print(json.dumps(report, indent=2, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"PMEmo feature-manifest error: {error}", file=sys.stderr)
        return 2
    return 0


def main(argv=None):
    parser = YAMLArgumentParser(prog="prepare_feature_manifest.py")
    configure_parser(parser)
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare DEAM feature or separation manifests from the supplied fixed splits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.configuration import YAMLArgumentParser, configure_yaml
from stemfusion.data.deam import build_feature_manifest
from stemfusion.data.deam_separation import DEAMSeparationManifestConfig, build_separation_manifest


_PRESETS = {
    "features": ("deam_feature_manifest", "experiments/deam/feature_manifest.yaml"),
    "separation": ("deam_separation_manifest", "experiments/deam/separation_manifest.yaml"),
}


def configure_parser(parser, *, purpose="features"):
    parser.add_argument("--purpose", choices=tuple(_PRESETS), default=purpose,
                        help="Manifest purpose; default: %(default)s. Each purpose loads its own YAML preset.")
    for name in ("train-split", "validation-split", "test-split", "mixture-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    features = parser.add_argument_group("feature-manifest options")
    for name in ("annotations", "htdemucs-root"):
        features.add_argument(f"--{name}", type=Path)
    separation = parser.add_argument_group("separation-manifest options")
    separation.add_argument("--output-root", type=Path)
    selection = separation.add_mutually_exclusive_group()
    selection.add_argument("--track-list", type=Path, help="Text file with one canonical numeric DEAM song ID per line.")
    selection.add_argument("--all-tracks", action="store_true", help="Select all tracks in the supplied splits.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    configure_yaml(parser, *_PRESETS[purpose])
    next(action for action in parser._actions if action.dest == "config").help = (
        "YAML preset for the selected purpose. Defaults: features -> feature_manifest.yaml; "
        "separation -> separation_manifest.yaml, under configs/experiments/deam/."
    )
    parser.yaml_purposes = _PRESETS
    parser.set_defaults(command_handler=run_from_args)


def run_from_args(args):
    try:
        explicit = args.yaml_provenance["cli_overrides"]
        incompatible = ({"output_root", "track_list", "all_tracks"} if args.purpose == "features"
                        else {"annotations", "htdemucs_root"})
        unused = sorted(incompatible.intersection(explicit))
        if unused:
            options = ", ".join("--" + name.replace("_", "-") for name in unused)
            raise ValueError(f"Options not supported for --purpose {args.purpose}: {options}")
        split_files = {split: getattr(args, f"{split}_split").expanduser().resolve()
                       for split in ("train", "validation", "test")}
        if args.purpose == "features":
            report = build_feature_manifest(
                split_files=split_files,
                **{name: getattr(args, name).expanduser().resolve() for name in (
                    "annotations", "mixture_root", "htdemucs_root", "output")},
                overwrite=args.overwrite, dry_run=args.dry_run)
        else:
            report = build_separation_manifest(DEAMSeparationManifestConfig(
                split_files=split_files, mixture_root=args.mixture_root.expanduser().resolve(),
                output_root=args.output_root.expanduser().resolve(), output_path=args.output.expanduser().resolve(),
                track_list=args.track_list.expanduser().resolve() if args.track_list is not None else None,
                all_tracks=args.all_tracks and "track_list" not in explicit,
                overwrite=args.overwrite, dry_run=args.dry_run))
        print(json.dumps(report, indent=2, sort_keys=True))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"DEAM {args.purpose} manifest error: {error}", file=sys.stderr)
        return 2
    return 0


def main(argv=None):
    parser = YAMLArgumentParser(prog="prepare_manifest.py")
    configure_parser(parser)
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

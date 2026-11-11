"""Prepare Jamendo feature or separation manifests without changing the audio."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.configuration import YAMLArgumentParser, configure_yaml
from stemfusion.data.mtg_jamendo import (
    JamendoSeparationManifestConfig,
    build_feature_manifest,
    build_separation_manifest,
)


_PRESETS = {
    "features": ("mtg_jamendo_feature_manifest", "experiments/mtg_jamendo/feature_manifest.yaml"),
    "separation": ("mtg_jamendo_separation_manifest", "experiments/mtg_jamendo/separation_manifest.yaml"),
}


def configure_parser(parser, *, purpose: str = "features") -> None:
    parser.add_argument("--purpose", choices=tuple(_PRESETS), default=purpose,
                        help="Manifest purpose; default: %(default)s. Each purpose loads its existing YAML preset.")
    for name in ("train-split", "validation-split", "test-split", "mixture-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    features = parser.add_argument_group("feature-manifest options")
    features.add_argument("--htdemucs-root", type=Path)
    features.add_argument("--audio-overrides", type=Path)
    features.add_argument("--no-audio-overrides", dest="audio_overrides", action="store_const", const=None,
                          help="Use only the original audio roots without a validated repair mapping.")
    separation = parser.add_argument_group("separation-manifest options")
    separation.add_argument("--output-root", type=Path)
    selection = separation.add_mutually_exclusive_group()
    selection.add_argument("--track-list", type=Path, help="Text file containing one canonical track_ID per line.")
    selection.add_argument("--all-tracks", action="store_true", help="Select the entire provided mood/theme split set.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    configure_yaml(parser, *_PRESETS[purpose])
    next(action for action in parser._actions if action.dest == "config").help = (
        "YAML preset for the selected purpose. Defaults: features -> feature_manifest.yaml; "
        "separation -> separation_manifest.yaml, under configs/experiments/mtg_jamendo/."
    )
    parser.yaml_purposes = _PRESETS
    parser.set_defaults(command_handler=run_from_args)


def run_from_args(args) -> int:
    try:
        explicit = args.yaml_provenance["cli_overrides"]
        incompatible = ({"output_root", "track_list", "all_tracks"} if args.purpose == "features"
                        else {"htdemucs_root", "audio_overrides"})
        unused = sorted(incompatible.intersection(explicit))
        if unused:
            options = ", ".join("--" + name.replace("_", "-") for name in unused)
            raise ValueError(f"Options not supported for --purpose {args.purpose}: {options}")
        split_files = {split: getattr(args, f"{split}_split").expanduser().resolve()
                       for split in ("train", "validation", "test")}
        if args.purpose == "features":
            result = build_feature_manifest(
                split_files=split_files, mixture_root=args.mixture_root.expanduser().resolve(),
                htdemucs_root=args.htdemucs_root.expanduser().resolve(),
                output=args.output.expanduser().resolve(),
                audio_overrides=args.audio_overrides.expanduser().resolve() if args.audio_overrides is not None else None,
                overwrite=args.overwrite, dry_run=args.dry_run)
        else:
            config = JamendoSeparationManifestConfig(
                split_files=split_files, mixture_root=args.mixture_root.expanduser().resolve(),
                output_root=args.output_root.expanduser().resolve(), output_path=args.output.expanduser().resolve(),
                track_list=args.track_list.expanduser().resolve() if args.track_list is not None else None,
                all_tracks=args.all_tracks, overwrite=args.overwrite, dry_run=args.dry_run)
            result = build_separation_manifest(config)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Jamendo {args.purpose} manifest error: {error}", file=sys.stderr)
        return 2
    return 0


def main(argv=None) -> int:
    parser = YAMLArgumentParser(prog="prepare_manifest.py")
    configure_parser(parser)
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

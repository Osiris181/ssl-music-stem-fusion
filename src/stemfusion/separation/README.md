# Source separation

## Purpose

This package contains wrappers around external music source-separation
systems. It does not vendor their model implementations or checkpoints.

## Code

| File | Command | Purpose |
| --- | --- | --- |
| `htdemucs.py` | `stemfusion separate htdemucs` | Extract all six `htdemucs_6s` sources and resume across storage locations. |
| `manifest.py` | Used by manifest-mode separation | Validate IDs, splits, and source paths independently of single- or multi-label targets. |

The wrapper makes effective inference settings explicit, validates outputs,
supports dry runs, and writes machine-readable reports. Original research
scripts remain outside this repository until output equivalence is checked.

## How to use it

Inspect the required model and path arguments, then begin with a dry run:

    stemfusion separate htdemucs --help

Full commands and output layouts are documented in
`docs/htdemucs_separation.md`.

FMA-Medium, GTZAN, ISMIR04, and MTG-Jamendo also expose `python separation.py` from their
experiment directories. These thin entry scripts load dataset-specific YAML
and call this same backend. Run `python separation.py --dry-run` first.

Directory mode discovers unique input basenames. Manifest mode (`--manifest`)
uses exact per-record stem paths beneath `--output-root`, skips complete native
six-stem sets, and stages temporary input symlinks for missing targets. This
preserves distinct recordings with identical filenames. It never edits the
manifest or source audio; dry runs create no staging/output/report files.

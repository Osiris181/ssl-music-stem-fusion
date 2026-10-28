# Source separation

## Purpose

This package contains wrappers around external music source-separation
systems. It does not vendor their model implementations or checkpoints.

## Code

| File | Command | Purpose |
| --- | --- | --- |
| `htdemucs.py` | `stemfusion separate htdemucs` | Extract all six `htdemucs_6s` sources and resume across storage locations. |

The wrapper makes effective inference settings explicit, validates outputs,
supports dry runs, and writes machine-readable reports. Original research
scripts remain outside this repository until output equivalence is checked.

## How to use it

Inspect the required model and path arguments, then begin with a dry run:

    stemfusion separate htdemucs --help

Full commands and output layouts are documented in
`docs/htdemucs_separation.md`.

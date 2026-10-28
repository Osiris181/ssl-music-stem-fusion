"""Batch dataset separation using the HT-Demucs command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = (".mp3",)
HTDEMUCS_6S_STEMS = (
    "bass",
    "drums",
    "guitar",
    "other",
    "piano",
    "vocals",
)


class HTDemucsConfigurationError(ValueError):
    """Raised when an HT-Demucs separation configuration is invalid."""


@dataclass(frozen=True)
class HTDemucsSeparationConfig:
    """Parameters for a reproducible HT-Demucs dataset separation run."""

    input_roots: tuple[Path, ...]
    output_root: Path
    existing_output_roots: tuple[Path, ...] = ()
    python_executable: str = sys.executable
    model_name: str = "htdemucs_6s"
    expected_stems: tuple[str, ...] = HTDEMUCS_6S_STEMS
    batch_size: int = 128
    device: str = "cuda"
    mp3_bitrate: int = 320
    mp3_preset: int = 2
    shifts: int = 1
    overlap: float = 0.25
    jobs: int = 0
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    recursive: bool = True
    overwrite: bool = False
    continue_on_error: bool = True
    limit: int | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class HTDemucsSeparationFailure:
    input_path: str
    return_code: int | None
    reason: str


@dataclass
class HTDemucsSeparationSummary:
    discovered: int
    selected: int
    planned: int = 0
    skipped_existing: int = 0
    succeeded: int = 0
    rescue_attempts: int = 0
    failed_batches: int = 0
    failures: list[HTDemucsSeparationFailure] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return len(self.failures)

    @property
    def exit_code(self) -> int:
        return 1 if self.failures else 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failed"] = self.failed
        data["exit_code"] = self.exit_code
        return data


def _normalize_extensions(extensions: Sequence[str]) -> tuple[str, ...]:
    normalized = {
        extension.lower()
        if extension.startswith(".")
        else f".{extension.lower()}"
        for extension in extensions
    }
    return tuple(sorted(normalized))


def validate_config(config: HTDemucsSeparationConfig) -> None:
    if not config.input_roots:
        raise HTDemucsConfigurationError(
            "At least one input root must be provided."
        )
    for input_root in config.input_roots:
        if not input_root.is_dir():
            raise HTDemucsConfigurationError(
                f"Input root is not a directory: {input_root}"
            )
    for existing_root in config.existing_output_roots:
        if not existing_root.is_dir():
            raise HTDemucsConfigurationError(
                f"Existing output root is not a directory: {existing_root}"
            )
    if config.model_name != "htdemucs_6s":
        raise HTDemucsConfigurationError(
            "The paper reproduction wrapper currently supports only "
            "the htdemucs_6s model."
        )
    if config.expected_stems != HTDEMUCS_6S_STEMS:
        raise HTDemucsConfigurationError(
            "Expected stems must match the six htdemucs_6s outputs."
        )
    if config.batch_size <= 0:
        raise HTDemucsConfigurationError("Batch size must be positive.")
    if config.mp3_bitrate <= 0:
        raise HTDemucsConfigurationError("MP3 bitrate must be positive.")
    if config.mp3_preset not in range(2, 8):
        raise HTDemucsConfigurationError("MP3 preset must be between 2 and 7.")
    if config.shifts < 0:
        raise HTDemucsConfigurationError("Shifts cannot be negative.")
    if not 0.0 <= config.overlap < 1.0:
        raise HTDemucsConfigurationError(
            "Overlap must be greater than or equal to 0 and less than 1."
        )
    if config.jobs < 0:
        raise HTDemucsConfigurationError("Jobs cannot be negative.")
    if config.limit is not None and config.limit <= 0:
        raise HTDemucsConfigurationError(
            "Limit must be positive when provided."
        )
    if not config.extensions:
        raise HTDemucsConfigurationError(
            "At least one input extension is required."
        )

    output_root = config.output_root.resolve()
    for input_root in config.input_roots:
        resolved_input = input_root.resolve()
        if output_root == resolved_input or output_root.is_relative_to(
            resolved_input
        ):
            raise HTDemucsConfigurationError(
                "Output root must not be an input root or one of its "
                "descendants."
            )


def discover_audio_files(config: HTDemucsSeparationConfig) -> list[Path]:
    """Discover all configured audio inputs in deterministic order."""

    extensions = set(_normalize_extensions(config.extensions))
    discovered: list[Path] = []
    for input_root in config.input_roots:
        candidates = (
            input_root.rglob("*")
            if config.recursive
            else input_root.glob("*")
        )
        discovered.extend(
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in extensions
        )

    discovered.sort(key=lambda path: str(path))
    duplicate_stems: dict[str, list[Path]] = defaultdict(list)
    for path in discovered:
        duplicate_stems[path.stem].append(path)
    collisions = {
        stem: paths
        for stem, paths in duplicate_stems.items()
        if len(paths) > 1
    }
    if collisions:
        details = "; ".join(
            f"{stem}: {', '.join(str(path) for path in paths)}"
            for stem, paths in sorted(collisions.items())
        )
        raise HTDemucsConfigurationError(
            "Demucs uses the input filename stem as its output directory, "
            f"so duplicate track stems are not allowed: {details}"
        )
    return discovered


def track_output_directory(
    output_root: Path,
    model_name: str,
    input_path: Path,
) -> Path:
    return output_root / model_name / input_path.stem


def is_complete_track(
    output_root: Path,
    config: HTDemucsSeparationConfig,
    input_path: Path,
    modified_since_ns: int | None = None,
) -> bool:
    track_directory = track_output_directory(
        output_root,
        config.model_name,
        input_path,
    )
    for stem in config.expected_stems:
        stem_path = track_directory / f"{stem}.mp3"
        if not stem_path.is_file() or stem_path.stat().st_size == 0:
            return False
        if (
            modified_since_ns is not None
            and stem_path.stat().st_mtime_ns < modified_since_ns
        ):
            return False
    return True

def _completion_roots(
    config: HTDemucsSeparationConfig,
) -> tuple[Path, ...]:
    roots = [config.output_root, *config.existing_output_roots]
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return tuple(unique)


def _is_complete_anywhere(
    config: HTDemucsSeparationConfig,
    input_path: Path,
) -> bool:
    return any(
        is_complete_track(root, config, input_path)
        for root in _completion_roots(config)
    )


def build_command(
    config: HTDemucsSeparationConfig,
    input_paths: Sequence[Path],
) -> list[str]:
    """Build a Demucs 4.0.1 command with explicit effective settings."""

    if not input_paths:
        raise ValueError("At least one input path is required.")
    return [
        config.python_executable,
        "-m",
        "demucs.separate",
        "--name",
        config.model_name,
        "--out",
        str(config.output_root),
        "--device",
        config.device,
        "--shifts",
        str(config.shifts),
        "--overlap",
        str(config.overlap),
        "--clip-mode",
        "rescale",
        "--mp3",
        "--mp3-bitrate",
        str(config.mp3_bitrate),
        "--mp3-preset",
        str(config.mp3_preset),
        "--jobs",
        str(config.jobs),
        *(str(path) for path in input_paths),
    ]


def _batches(paths: Sequence[Path], size: int) -> Iterable[list[Path]]:
    for start in range(0, len(paths), size):
        yield list(paths[start : start + size])


def _failure(
    input_path: Path,
    return_code: int | None,
    reason: str,
) -> HTDemucsSeparationFailure:
    return HTDemucsSeparationFailure(
        input_path=str(input_path),
        return_code=return_code,
        reason=reason,
    )


def _run_command(
    config: HTDemucsSeparationConfig,
    input_paths: Sequence[Path],
    runner: Callable[..., Any],
    capture_output: bool,
) -> tuple[int | None, str | None, int]:
    started_ns = time.time_ns()
    try:
        result = runner(
            build_command(config, input_paths),
            check=False,
            capture_output=capture_output,
        )
        return int(result.returncode), None, started_ns
    except OSError as error:
        return None, str(error), started_ns


def separate_dataset(
    config: HTDemucsSeparationConfig,
    runner: Callable[..., Any] = subprocess.run,
) -> HTDemucsSeparationSummary:
    """Separate configured tracks, retrying failed batches track by track."""

    validate_config(config)
    discovered = discover_audio_files(config)
    if not discovered:
        extensions = ", ".join(config.extensions)
        raise HTDemucsConfigurationError(
            "No input audio found in the configured roots for extensions: "
            f"{extensions}"
        )

    selected = discovered[: config.limit] if config.limit else discovered
    summary = HTDemucsSeparationSummary(
        discovered=len(discovered),
        selected=len(selected),
    )
    pending: list[Path] = []
    for input_path in selected:
        if not config.overwrite and _is_complete_anywhere(config, input_path):
            summary.skipped_existing += 1
        else:
            pending.append(input_path)

    if config.dry_run:
        summary.planned = len(pending)
        for input_path in pending:
            LOGGER.info("Would separate %s", input_path)
        return summary

    config.output_root.mkdir(parents=True, exist_ok=True)
    total_batches = (len(pending) + config.batch_size - 1) // config.batch_size
    for batch_number, batch in enumerate(
        _batches(pending, config.batch_size),
        start=1,
    ):
        LOGGER.info(
            "Processing HT-Demucs batch %d/%d with %d tracks.",
            batch_number,
            total_batches,
            len(batch),
        )
        return_code, execution_error, started_ns = _run_command(
            config,
            batch,
            runner,
            capture_output=False,
        )

        if execution_error is not None:
            summary.failed_batches += 1
            summary.failures.extend(
                _failure(input_path, None, execution_error)
                for input_path in batch
            )
            break

        incomplete: list[Path] = []
        for input_path in batch:
            modified_since = started_ns if config.overwrite else None
            if is_complete_track(
                config.output_root,
                config,
                input_path,
                modified_since_ns=modified_since,
            ):
                summary.succeeded += 1
            else:
                incomplete.append(input_path)

        if not incomplete:
            continue

        if return_code != 0:
            summary.failed_batches += 1
            LOGGER.warning(
                "Batch %d exited with status %d; retrying %d incomplete "
                "tracks individually.",
                batch_number,
                return_code,
                len(incomplete),
            )
        else:
            LOGGER.warning(
                "Batch %d exited successfully but left %d incomplete tracks; "
                "retrying them individually.",
                batch_number,
                len(incomplete),
            )

        if len(batch) == 1:
            reason = (
                f"Demucs exited with status {return_code}."
                if return_code != 0
                else "Demucs did not produce all six non-empty stem files."
            )
            summary.failures.append(
                _failure(incomplete[0], return_code, reason)
            )
            if not config.continue_on_error:
                return summary
            continue

        for input_path in incomplete:
            summary.rescue_attempts += 1
            rescue_code, rescue_error, rescue_started_ns = _run_command(
                config,
                [input_path],
                runner,
                capture_output=True,
            )
            modified_since = rescue_started_ns if config.overwrite else None
            if (
                rescue_error is None
                and rescue_code == 0
                and is_complete_track(
                    config.output_root,
                    config,
                    input_path,
                    modified_since_ns=modified_since,
                )
            ):
                summary.succeeded += 1
                continue

            if rescue_error is not None:
                reason = rescue_error
            elif rescue_code != 0:
                reason = f"Demucs rescue exited with status {rescue_code}."
            else:
                reason = (
                    "Demucs rescue did not produce all six non-empty "
                    "stem files."
                )
            summary.failures.append(
                _failure(input_path, rescue_code, reason)
            )
            LOGGER.error("%s Input: %s", reason, input_path)
            if not config.continue_on_error:
                return summary

    return summary


def write_report(
    summary: HTDemucsSeparationSummary,
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Separate datasets into six stems using the paper's HT-Demucs setup."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        required=True,
        help="Input root; may be specified more than once.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--existing-output-root",
        type=Path,
        action="append",
        default=[],
        help="Read-only completed-output root; may be repeated.",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--model-name", default="htdemucs_6s")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mp3-bitrate", type=int, default=320)
    parser.add_argument("--mp3-preset", type=int, default=2)
    parser.add_argument("--shifts", type=int, default=1)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument(
        "--extension",
        action="append",
        dest="extensions",
        help="Input extension; may be repeated. Default: .mp3",
    )
    parser.add_argument("--non-recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.set_defaults(command_handler=run_from_args)


def config_from_args(args: argparse.Namespace) -> HTDemucsSeparationConfig:
    return HTDemucsSeparationConfig(
        input_roots=tuple(
            path.expanduser().resolve() for path in args.input_root
        ),
        output_root=args.output_root.expanduser().resolve(),
        existing_output_roots=tuple(
            path.expanduser().resolve()
            for path in args.existing_output_root
        ),
        python_executable=args.python_executable,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        mp3_bitrate=args.mp3_bitrate,
        mp3_preset=args.mp3_preset,
        shifts=args.shifts,
        overlap=args.overlap,
        jobs=args.jobs,
        extensions=_normalize_extensions(args.extensions or DEFAULT_EXTENSIONS),
        recursive=not args.non_recursive,
        overwrite=args.overwrite,
        continue_on_error=not args.fail_fast,
        limit=args.limit,
        dry_run=args.dry_run,
    )


def run_from_args(args: argparse.Namespace) -> int:
    try:
        config = config_from_args(args)
        summary = separate_dataset(config)
    except HTDemucsConfigurationError as error:
        LOGGER.error("%s", error)
        return 2

    report_path = args.report
    if not config.dry_run:
        if report_path is None:
            report_path = config.output_root / "separation_report_htdemucs.json"
        write_report(summary, report_path.expanduser().resolve())

    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return summary.exit_code

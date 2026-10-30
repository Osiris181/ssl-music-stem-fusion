"""Strict YAML configuration loading for public command-line tasks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """A configuration cannot be interpreted by the current implementation."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects accidentally repeated mapping keys."""


def _mapping(loader: _UniqueKeyLoader, node: Any) -> dict[str, Any]:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            raise ConfigurationError("YAML mapping keys must be strings.")
        if key in result:
            raise ConfigurationError(f"Duplicate YAML key: {key}.")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping
)

# Configurable values are mapped explicitly to argparse destinations.
_EXTRACTION = {
    "extractor.checkpoint": ("checkpoint_filename", "str"),
    "extractor.statistics": ("statistics_filename", "str"),
    "audio.sample_rate_hz": ("sample_rate", "int"),
    "audio.max_seconds": ("max_seconds", "optional_float"),
    "features.conformer_layer_index": ("layer_index", "int"),
    "features.temporal_pooling": ("temporal_pooling", "str"),
    "features.dimension": ("expected_dimension", "int"),
    "runtime.device": ("device", "str"),
    "runtime.flash_attention": ("is_flash", "bool"),
    "output.files_per_source": ("splits", "str_list"),
}
_TRAINING = {
    "features.dimension_per_source": ("expected_feature_dimension", "int"),
    "model.hidden_dimensions": ("hidden_dimensions", "int_list"),
    "model.dropout": ("dropout", "float"),
    "optimization.learning_rate": ("learning_rate", "float"),
    "optimization.batch_size": ("batch_size", "int"),
    "optimization.epochs": ("epochs", "int"),
    "reproducibility.seeds": ("seeds", "int_list"),
    "runtime.device": ("device", "str"),
    "runtime.num_workers": ("num_workers", "int"),
}
_SEPARATOR = {
    "separator.model_name": ("model_name", "str"),
    "inference.device": ("device", "str"),
    "inference.shifts": ("shifts", "int"),
    "inference.overlap": ("overlap", "float"),
    "inference.jobs": ("jobs", "int"),
    "output.bitrate_kbps": ("mp3_bitrate", "int"),
    "output.mp3_preset": ("mp3_preset", "int"),
    "batching.default_tracks_per_process": ("batch_size", "int"),
    "inputs.default_extensions": ("extensions", "str_list"),
}
_SOURCE_ORDER = ["mixture", "guitar", "drums", "piano", "bass", "vocals"]
_EXTRACTION_FIXED = {
    "extractor.name": "musicfm",
    "extractor.architecture": "MusicFM25Hz",
    "extractor.frozen": True,
    "audio.channels": "mono",
    "features.dtype": "float32",
    "output.format": "pickle",
    "output.key": "zero_padded_fma_track_id",
}
_TRAINING_FIXED = {
    "task.name": "genre_classification",
    "task.label_type": "multiclass",
    "task.number_of_classes": 16,
    "features.name": "musicfm_msd_layer7",
    "features.source_order": _SOURCE_ORDER,
    "model.activation": "relu",
    "optimization.loss": "cross_entropy",
    "optimization.optimizer": "adam",
    "optimization.betas": [0.9, 0.999],
    "optimization.epsilon": 1e-8,
    "optimization.weight_decay": 0.0,
    "optimization.scheduler": "none",
    "optimization.early_stopping": "none",
    "checkpoint.split": "validation",
    "checkpoint.maximize": ["accuracy_micro", "roc_auc_macro", "pr_auc_macro"],
    "checkpoint.aggregation": "sum",
    "checkpoint.ties": "earliest_epoch",
    "evaluation.accuracy": "micro",
    "evaluation.precision": "macro",
    "evaluation.recall": "macro",
    "evaluation.f1": "macro",
    "evaluation.roc_auc": "macro_one_vs_rest",
    "evaluation.pr_auc": "macro_average_precision",
    "evaluation.absent_classes": "error",
    "features.silent_vector_policy": "retain",
    "features.normalization": "none",
    "reproducibility.cudnn_deterministic": True,
    "reproducibility.cudnn_benchmark": False,
}
_SEPARATOR_FIXED = {
    "separator.name": "htdemucs",
    "separator.package": "demucs",
    "separator.package_version": "4.0.1",
    "separator.expected_stems": ["bass", "drums", "guitar", "other", "piano", "vocals"],
    "inference.split": True,
    "inference.clip_mode": "rescale",
    "output.format": "mp3",
    "output.filename_template": "{track}/{stem}.{ext}",
}
_METADATA = {
    "dataset", "task", "number_of_classes", "separator", "command",
    "implementation_status", "sources", "source", "runs",
    "suggested_output_directory",
}
_PATHS = {
    "extraction": {"manifest", "output_dir", "musicfm_root", "stats_path", "checkpoint_path", "report"},
    "mid_fusion": {"manifest", "feature_dir", "output_dir", "summary_dir"},
    "late_fusion": {"manifest", "feature_dir", "output_dir", "summary_dir"},
    "separation": {"input_root", "output_root", "existing_output_root", "report"},
}
_EXECUTION = {
    "extraction": {
        "limit_per_split": ("limit_per_split", "int"),
        "overwrite": ("overwrite", "bool"),
        "fail_fast": ("fail_fast", "bool"),
        "dry_run": ("dry_run", "bool"),
    },
    "mid_fusion": {
        "overwrite": ("overwrite", "bool"),
        "resume": ("resume", "bool"),
        "validate_only": ("validate_only", "bool"),
        "dry_run": ("dry_run", "bool"),
    },
    "late_fusion": {
        "overwrite": ("overwrite", "bool"),
        "resume": ("resume", "bool"),
        "validate_only": ("validate_only", "bool"),
        "dry_run": ("dry_run", "bool"),
    },
    "separation": {
        "non_recursive": ("non_recursive", "bool"),
        "overwrite": ("overwrite", "bool"),
        "fail_fast": ("fail_fast", "bool"),
        "limit": ("limit", "int"),
        "dry_run": ("dry_run", "bool"),
        "python_executable": ("python_executable", "str"),
    },
}


def default_config_path(relative: str) -> Path:
    """Find the one canonical preset tree in a checkout or installed package."""

    roots = (
        Path(__file__).resolve().parents[2] / "configs",
        Path(sys.prefix) / "share" / "stemfusion" / "configs",
    )
    for root in roots:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    raise ConfigurationError(
        "Cannot find packaged configuration files. Use an explicit --config "
        "path or install the project with its configuration data."
    )


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, name))
        else:
            result[name] = value
    return result


def _coerce(value: Any, kind: str, name: str) -> Any:
    if kind == "optional_float" and value is None:
        return None
    if kind in {"int_list", "str_list"}:
        if not isinstance(value, list) or not value:
            raise ConfigurationError(f"{name} must be a nonempty list.")
        element = "int" if kind == "int_list" else "str"
        return [_coerce(item, element, name) for item in value]
    if kind == "bool" and type(value) is bool:
        return value
    if kind == "int" and type(value) is int:
        return value
    if kind == "str" and isinstance(value, str) and value:
        return value
    if kind in {"float", "optional_float"} and type(value) in {int, float, str}:
        try:
            converted = float(value)
        except ValueError:
            pass
        else:
            if math.isfinite(converted):
                return converted
    raise ConfigurationError(f"{name} has an invalid value/type: {value!r}.")


def _resolve_path(value: str, parent: Path) -> Path:
    path = Path(_coerce(value, "str", "path")).expanduser()
    return (parent / path).resolve() if not path.is_absolute() else path.resolve()


def load_configuration(
    path: Path, role: str, run_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a preset or experiment with relative profile/path resolution."""

    files: dict[str, str] = {}

    def read(current: Path) -> dict[str, Any]:
        try:
            raw = current.read_bytes()
            data = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeyLoader)
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ConfigurationError(f"Cannot read YAML {current}: {error}") from error
        if not isinstance(data, dict):
            raise ConfigurationError(f"{current} must contain a YAML mapping.")
        files[str(current)] = hashlib.sha256(raw).hexdigest()
        return data

    def load(current: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
        current = current.expanduser().resolve()
        if current in stack:
            raise ConfigurationError(f"Cyclic configuration reference: {current}")
        data = read(current)
        if isinstance(data.get("task"), str):
            if data.pop("task") != "genre_classification":
                raise ConfigurationError("Only genre_classification is supported.")
        profile_key = {
            "extraction": "feature_profile",
            "separation": "separator_profile",
        }.get(role, "training_profile")
        profile = data.pop(profile_key, None)
        base = load(_resolve_path(profile, current.parent), (*stack, current)) if profile else {}
        # Ancillary profiles describe the already-produced audio/features.
        # They are recorded, but extraction never launches separation.
        for key in ("feature_profile", "separator_profile"):
            reference = data.pop(key, None)
            if reference is not None:
                read(_resolve_path(reference, current.parent))
        paths = data.get("paths", {})
        if not isinstance(paths, dict):
            raise ConfigurationError("paths must be a mapping.")
        resolved_paths = {}
        for key, value in paths.items():
            if key not in _PATHS[role]:
                raise ConfigurationError(f"Unsupported path setting: paths.{key}")
            if value is not None:
                if key in {"input_root", "existing_output_root"}:
                    values = value if isinstance(value, list) else [value]
                    resolved_paths[key] = [_resolve_path(item, current.parent) for item in values]
                else:
                    resolved_paths[key] = _resolve_path(value, current.parent)
        if "paths" in data:
            data["paths"] = resolved_paths
        return _merge(base, data)

    data = load(path, ())
    yaml_snapshot = json.loads(json.dumps(data, default=str))
    defaults: dict[str, Any] = {}
    mappings = dict(_EXTRACTION if role == "extraction" else _SEPARATOR if role == "separation" else _TRAINING)
    fixed = dict(_EXTRACTION_FIXED if role == "extraction" else _SEPARATOR_FIXED if role == "separation" else _TRAINING_FIXED)
    if role == "mid_fusion":
        fixed.update({"model.name": "deep_mlp", "features.fusion": "concatenate"})
    elif role == "late_fusion":
        fixed.update({"model.name": "per_source_deep_mlp", "features.fusion": "independent_logits_then_equal_sum"})

    metadata = {}
    if isinstance(data.get("task"), str):
        if data["task"] != "genre_classification":
            raise ConfigurationError("Only the genre_classification task is supported.")
        metadata["task"] = data.pop("task")
    for key in _METADATA - {"task"}:
        if key in data and not (key == "separator" and isinstance(data[key], dict)):
            metadata[key] = data.pop(key)
    expected_command = {
        "extraction": "extract fma-medium",
        "mid_fusion": "train fma-medium mid-fusion",
        "late_fusion": "train fma-medium late-fusion",
    }.get(role)
    if metadata.get("command", expected_command) != expected_command:
        raise ConfigurationError(f"Configuration command does not match {expected_command}.")
    if metadata.get("dataset", "fma_medium") != "fma_medium":
        raise ConfigurationError("Only FMA-Medium dataset configurations are supported.")
    if metadata.get("number_of_classes", 16) != 16:
        raise ConfigurationError("FMA-Medium configurations must describe 16 classes.")
    for name in ("implementation_status", "suggested_output_directory", "source", "separator"):
        if name in metadata:
            _coerce(metadata[name], "str", name)
    if "sources" in metadata:
        _coerce(metadata["sources"], "str_list", "sources")

    runs = metadata.get("runs", {})
    if not isinstance(runs, dict):
        raise ConfigurationError("runs must be a mapping.")
    for name, condition in runs.items():
        if not isinstance(condition, dict) or "sources" not in condition or set(condition) - {"sources", "kind", "matched_sources"}:
            raise ConfigurationError(f"Run {name!r} must contain sources and optional kind/matched_sources.")
        _coerce(condition["sources"], "str_list", f"runs.{name}.sources")
        if "kind" in condition:
            _coerce(condition["kind"], "str", f"runs.{name}.kind")
        if "matched_sources" in condition:
            _coerce(condition["matched_sources"], "int", f"runs.{name}.matched_sources")
    if run_name is not None:
        if role not in {"mid_fusion", "late_fusion"} or run_name not in runs:
            raise ConfigurationError(f"Unknown/inapplicable run {run_name!r}; available: {', '.join(runs)}")
        selected = runs[run_name]
        defaults["source"] = _coerce(selected["sources"], "str_list", "runs.sources")
    elif role in {"mid_fusion", "late_fusion"} and "sources" in metadata:
        defaults["source"] = _coerce(metadata["sources"], "str_list", "sources")
    elif role == "extraction":
        if "source" in metadata:
            defaults["source"] = [_coerce(metadata["source"], "str", "source")]
        elif "sources" in metadata:
            defaults["source"] = _coerce(metadata["sources"], "str_list", "sources")

    paths = data.pop("paths", {})
    for name, value in paths.items():
        defaults[name] = value
    execution = data.pop("execution", {})
    if not isinstance(execution, dict):
        raise ConfigurationError("execution must be a mapping.")
    for name, value in execution.items():
        if name not in _EXECUTION[role]:
            raise ConfigurationError(f"Unsupported execution setting: {name}")
        destination, kind = _EXECUTION[role][name]
        defaults[destination] = _coerce(value, kind, f"execution.{name}")

    # Historical annotations are not active FMA settings.
    if role == "separation":
        for section, key in (("batching", "ismir04_tracks_per_process"), ("inputs", "ismir04_extensions")):
            data.get(section, {}).pop(key, None)
    flat = _flatten(data)
    for name in set(flat) - set(mappings) - set(fixed):
        raise ConfigurationError(f"Unknown/unsupported YAML setting: {name}")
    for name, (destination, kind) in mappings.items():
        if name not in flat:
            raise ConfigurationError(f"Missing required YAML setting: {name}")
        defaults[destination] = _coerce(flat[name], kind, name)
    for name, expected in fixed.items():
        if name not in flat or flat[name] != expected or type(flat[name]) is not type(expected):
            raise ConfigurationError(
                f"{name} must be {expected!r}; other values are not implemented."
            )
    if role == "extraction":
        for name in ("checkpoint_filename", "statistics_filename"):
            if Path(defaults[name]).name != defaults[name]:
                raise ConfigurationError(f"{name} must be a basename; use paths for explicit file locations.")
    provenance = {
        "config_files": [{"path": name, "sha256": sha} for name, sha in files.items()],
        "run_name": run_name,
        "experiment_metadata": metadata,
        "yaml_settings_before_cli_overrides": yaml_snapshot,
        "cli_overrides": {},
    }
    return defaults, provenance


def configuration_dict(config: Any) -> dict[str, Any]:
    """Serialize a resolved dataclass configuration, including paths."""

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [convert(item) for item in value]
        return value

    return convert(asdict(config))


class YAMLArgumentParser(argparse.ArgumentParser):
    """Resolve YAML defaults before parsing a public task's CLI overrides."""

    yaml_role: str | None = None
    yaml_default: str | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def parse_known_args(self, args: Any = None, namespace: Any = None) -> Any:
        if self.yaml_role is None:
            return super().parse_known_args(args, namespace)
        arguments = list(sys.argv[1:] if args is None else args)
        if any(option in arguments for option in ("-h", "--help")):
            return super().parse_known_args(arguments, namespace)
        selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        selector.add_argument("--config", type=Path)
        if self.yaml_role in {"mid_fusion", "late_fusion"}:
            selector.add_argument("--run")
        selected, _ = selector.parse_known_args(arguments)
        try:
            path = selected.config or default_config_path(self.yaml_default)
            defaults, provenance = load_configuration(
                path, self.yaml_role, getattr(selected, "run", None)
            )
            action_map = {action.dest: action for action in self._actions}
            explicit = set()
            for action in self._actions:
                if any(token.split("=", 1)[0] in action.option_strings for token in arguments):
                    explicit.add(action.dest)
            for name, value in defaults.items():
                action = action_map.get(name)
                if action is not None and action.choices is not None:
                    values = value if isinstance(value, list) else [value]
                    if any(item not in action.choices for item in values):
                        raise ConfigurationError(f"Invalid configured choice for {name}: {value!r}")
                if action is not None and isinstance(action, argparse._AppendAction) and name in explicit:
                    defaults[name] = None
            old_defaults = self._defaults.copy()
            old_actions = [(action, action.default, action.required) for action in self._actions]
            try:
                self.set_defaults(**copy.deepcopy(defaults))
                for action in self._actions:
                    if action.dest in defaults and defaults[action.dest] is not None:
                        action.required = False
                parsed, extras = super().parse_known_args(arguments, namespace)
                provenance["cli_overrides"] = json.loads(json.dumps({
                    name: getattr(parsed, name)
                    for name in explicit if name not in {"config", "run", "help"}
                }, default=str))
                parsed.yaml_provenance = provenance
                return parsed, extras
            finally:
                self._defaults = old_defaults
                for action, default, required in old_actions:
                    action.default, action.required = default, required
        except ConfigurationError as error:
            self.error(str(error))


def configure_yaml(parser: argparse.ArgumentParser, role: str, preset: str) -> None:
    """Attach configuration options to a dataset/separator task parser."""

    parser.add_argument(
        "--config", type=Path,
        help=f"YAML preset or experiment file. Default preset: configs/{preset}",
    )
    if role in {"mid_fusion", "late_fusion"}:
        parser.add_argument("--run", help="Run one named condition; omit to run the complete YAML matrix.")
    if not isinstance(parser, YAMLArgumentParser):
        raise TypeError("Public YAML-configured tasks require YAMLArgumentParser.")
    parser.yaml_role = role
    parser.yaml_default = preset

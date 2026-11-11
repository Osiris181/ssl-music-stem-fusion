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
    "audio.chunk_seconds": ("chunk_seconds", "optional_float"),
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
JAMENDO_MID_CONDITIONS = {
    "mixture_only": {"kind": "reference", "sources": ["mixture"]},
    **{f"repeated_mixture_{count}": {"kind": "repeated_mixture", "sources": ["mixture"], "matched_sources": count}
       for count in (4, 5, 6)},
    "mixture_guitar_drums_piano": {"kind": "fusion", "sources": _SOURCE_ORDER[:4]},
    "mixture_guitar_drums_piano_bass": {"kind": "fusion", "sources": _SOURCE_ORDER[:5]},
    "mixture_all_five_stems": {"kind": "fusion", "sources": list(_SOURCE_ORDER)},
}
JAMENDO_LATE_CONDITIONS = {
    "mixture_only": {"kind": "reference", "sources": ["mixture"]},
    **{f"mixture_ensemble_{count}": {"kind": "mixture_ensemble", "sources": ["mixture"], "matched_sources": count}
       for count in (4, 5, 6)},
    "mixture_guitar_drums_piano": {"kind": "fusion", "sources": _SOURCE_ORDER[:4]},
    "mixture_guitar_drums_piano_bass": {"kind": "fusion", "sources": _SOURCE_ORDER[:5]},
    "mixture_all_five_stems": {"kind": "fusion", "sources": list(_SOURCE_ORDER)},
    **{f"{source}_only": {"kind": "stem_only", "sources": [source]} for source in _SOURCE_ORDER[1:]},
}
_EXTRACTION_FIXED = {
    "extractor.name": "musicfm",
    "extractor.architecture": "MusicFM25Hz",
    "extractor.frozen": True,
    "audio.channels": "mono",
    "features.dtype": "float32",
    "output.format": "pickle",
    "output.key": "track_id",
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
    "deam_mid_fusion": {"manifest", "mert_dir", "harmony_dir", "chord_root", "chord_attr", "output_dir", "summary_dir"},
    "emotion_late_fusion": {"manifest", "mert_dir", "harmony_dir", "chord_root", "chord_attr", "output_dir", "summary_dir", "reference_output_dir"},
    "emotion_mid_fusion": {"manifest", "mert_dir", "harmony_dir", "chord_root", "chord_attr", "output_dir", "summary_dir"},
    "jamendo_late_fusion": {"manifest", "mert_dir", "harmony_dir", "tag_list", "author_metadata", "output_dir", "summary_dir", "reference_output_dir"},
    "jamendo_mid_fusion": {"manifest", "mert_dir", "harmony_dir", "tag_list", "author_metadata", "output_dir", "summary_dir", "reference_output_dir"},
    "emomusic_feature_manifest": {"train_split", "validation_split", "test_split", "annotations", "mixture_root", "htdemucs_root", "output"},
    "deam_feature_manifest": {"train_split", "validation_split", "test_split", "annotations", "mixture_root", "htdemucs_root", "output"},
    "deam_separation_manifest": {"train_split", "validation_split", "test_split", "track_list", "mixture_root", "output_root", "output"},
    "pmemo_feature_manifest": {"train_split", "validation_split", "test_split", "annotations", "metadata", "mixture_root", "htdemucs_root", "output"},
    "pmemo_separation_manifest": {"train_split", "validation_split", "test_split", "track_list", "mixture_root", "output_root", "output"},
    "jamendo_sheetsage2_training": {"manifest", "mert_dir", "harmony_dir", "original_harmony_dir", "tag_list", "author_metadata", "baseline_summary", "output_dir", "summary_dir"},
    "sheetsage2_extraction": {"manifest", "model_dir", "backbone_dir", "output_dir", "report"},
    "jamendo_chordformer_training": {"manifest", "mert_dir", "harmony_dir", "skey_dir", "tag_list", "author_metadata", "baseline_summary", "skey_summary", "output_dir", "summary_dir"},
    "chordformer_extraction": {"manifest", "skey_dir", "chordformer_dir", "output_dir", "report"},
    "jamendo_skey_training": {"manifest", "mert_dir", "harmony_dir", "tag_list", "author_metadata", "baseline_summary", "output_dir", "summary_dir"},
    "skey_extraction": {"manifest", "btc_dir", "output_dir", "checkpoint", "report"},
    "jamendo_reproduction": {"manifest", "mert_dir", "harmony_dir", "tag_list", "author_metadata", "output_dir", "summary_dir"},
    "mert_segment_extraction": {"manifest", "output_dir", "model_path", "report"},
    "chord_key_extraction": {"manifest", "output_dir", "author_root", "btc_config", "btc_checkpoint", "report"},
    "mtg_jamendo_feature_manifest": {"train_split", "validation_split", "test_split", "mixture_root", "htdemucs_root", "audio_overrides", "output"},
    "mtg_jamendo_separation_manifest": {"train_split", "validation_split", "test_split", "track_list", "mixture_root", "output_root", "output"},
    "fma_manifest": {"tracks_csv", "mixture_root", "htdemucs_root", "output"},
    "ismir04_manifest": {"metadata_root", "audio_root", "htdemucs_root", "repair_root", "output", "audit_path"},
    "gtzan_manifest": {"train_split", "validation_split", "test_split", "mixture_root", "htdemucs_root", "output"},
    "extraction": {"manifest", "output_dir", "musicfm_root", "stats_path", "checkpoint_path", "report"},
    "mid_fusion": {"manifest", "feature_dir", "output_dir", "summary_dir"},
    "late_fusion": {"manifest", "feature_dir", "output_dir", "summary_dir"},
    "separation": {"input_root", "output_root", "existing_output_root", "manifest", "report"},
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
    if kind == "optional_int":
        if value is None:
            return None
        kind = "int"
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
            "mert_segment_extraction": "feature_profile",
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
                if key in {"input_root", "existing_output_root"} or (role == "fma_manifest" and key == "htdemucs_root"):
                    values = value if isinstance(value, list) else [value]
                    resolved_paths[key] = [_resolve_path(item, current.parent) for item in values]
                else:
                    resolved_paths[key] = _resolve_path(value, current.parent)
        if "paths" in data:
            data["paths"] = resolved_paths
        return _merge(base, data)

    data = load(path, ())
    yaml_snapshot = json.loads(json.dumps(data, default=str))
    if role in {"emotion_mid_fusion", "emotion_late_fusion", "deam_mid_fusion"}:
        defaults = dict(data.pop("paths", {}))
        runs = data.pop("runs", None)
        flat = _flatten(data)
        fixed = {
            "dataset": "pmemo", "author_commit": "c5f7984f78d9928b841f25bd05a47377c12c58ca",
            "task.name": "emotion_recognition", "task.label_type": "binary", "task.score_scale": "zero_one",
            "task.label_threshold": 0.5, "task.positive_comparison": "greater_equal",
            "features.saved_layer_indices": [5, 6], "features.audio_dimension": 1536,
            "features.training_pooling": "random_consecutive_segment_mean", "features.evaluation_pooling": "all_segment_mean",
            "features.normalization": "none", "features.unavailable_key_mode": "major",
            "features.fusion": "concatenate", "features.harmony_encoder": "shared", "features.training_alignment": "shared_segment_range",
            "model.architecture": "music2emo_single_task", "model.chord_sequence_length": 100,
            "model.root_vocabulary_size": 14, "model.quality_vocabulary_size": 14,
            "model.root_embedding_dimension": 4, "model.quality_embedding_dimension": 4,
            "model.transformer_layers": 2, "model.transformer_heads": 8, "model.transformer_feedforward_dimension": 64,
            "model.transformer_dropout": 0.1, "model.cls_token": "zeros", "model.padding_mask": "none",
            "model.hidden_dimensions": [512, 256], "model.activation": "relu", "model.output_labels": 1,
            "model.multitask": False, "model.knowledge_distillation": False,
            "optimization.optimizer": "adam", "optimization.loss": "author_frequency_weighted_bce",
            "optimization.betas": [0.9, 0.999], "optimization.epsilon": 1e-8, "optimization.weight_decay": 0.0,
            "optimization.early_stopping.monitor": "val_loss", "optimization.early_stopping.mode": "min",
            "optimization.scheduler": "none", "checkpoint.split": "validation",
            "checkpoint.minimize": "weighted_bce_loss", "checkpoint.ties": "earliest_epoch",
            "checkpoint.save_top_k": 5, "checkpoint.save_last": True,
            "evaluation.output_transform": "sigmoid", "evaluation.accuracy": "binary", "evaluation.f1": "macro_two_class",
            "evaluation.roc_auc": "macro_two_class_one_vs_rest", "evaluation.pr_auc": "macro_two_class_average_precision",
            "evaluation.threshold_policy": "validation_macro_f1_global_grid",
            "evaluation.threshold_grid.start": 0.0, "evaluation.threshold_grid.stop": 1.0, "evaluation.threshold_grid.steps": 101,
            "reproducibility.deterministic_algorithms": True, "reproducibility.sampling_seed": "sha256_per_seed_per_epoch",
            "runtime.matmul_precision": "medium", "runtime.attention_backend": "math"}
        mappings = {"task.targets": ("targets", "str_list"),
                    "optimization.learning_rate": ("learning_rate", "float"), "optimization.batch_size": ("batch_size", "int"),
                    "optimization.epochs": ("epochs", "int"), "reproducibility.seeds": ("seeds", "int_list"),
                    "optimization.early_stopping.patience": ("early_stopping_patience", "int"),
                    "optimization.early_stopping.min_delta": ("early_stopping_min_delta", "float"),
                    "runtime.device": ("device", "str"), "runtime.num_workers": ("num_workers", "int"),
                    "runtime.verify_checksums": ("verify_checksums", "bool"),
                    "execution.resume": ("resume", "bool"), "execution.overwrite": ("overwrite", "bool"),
                    "execution.validate_only": ("validate_only", "bool"), "execution.dry_run": ("dry_run", "bool")}
        if role == "deam_mid_fusion":
            fixed.update({"dataset": "deam", "task.score_scale": "one_nine", "task.label_threshold": 5.0})
        late_fusion = role == "emotion_late_fusion"
        if late_fusion:
            fixed.update({"features.fusion": "sum_logits_then_sigmoid",
                          "features.harmony_encoder": "independent_per_branch",
                          "features.training_alignment": "independent_per_branch",
                          "reproducibility.branch_seed": "sha256_position_paired"})
        if set(defaults) != _PATHS[role] or set(flat) != set(fixed) | set(mappings):
            raise ConfigurationError("Unknown or missing binary-emotion fusion settings.")
        for name, expected in fixed.items():
            if flat[name] != expected or type(flat[name]) is not type(expected):
                raise ConfigurationError(f"{name} must be {expected!r}.")
        expected_runs = JAMENDO_LATE_CONDITIONS if late_fusion else JAMENDO_MID_CONDITIONS
        if runs != expected_runs or (run_name is not None and run_name not in runs):
            raise ConfigurationError("Use the shared fixed fusion matrix and a valid run name.")
        for name, (destination, kind) in mappings.items():
            defaults[destination] = _coerce(flat[name], kind, name)
        defaults.update(runs=runs, run=run_name)
        return defaults, {"config_files": [{"path": name, "sha256": sha} for name, sha in files.items()],
                          "yaml_settings_before_cli_overrides": yaml_snapshot, "cli_overrides": {}}
    if role in {"jamendo_reproduction", "jamendo_mid_fusion", "jamendo_late_fusion", "jamendo_skey_training", "jamendo_chordformer_training", "jamendo_sheetsage2_training"}:
        defaults = dict(data.pop("paths", {}))
        fusion_role = role in {"jamendo_mid_fusion", "jamendo_late_fusion"}
        runs = data.pop("runs", None) if fusion_role else None
        flat = _flatten(data)
        fixed = {
            "dataset": "mtg_jamendo", "author_commit": "c5f7984f78d9928b841f25bd05a47377c12c58ca",
            "features.source": "mixture", "features.saved_layer_indices": [5, 6], "features.audio_dimension": 1536,
            "features.training_pooling": "random_consecutive_segment_mean", "features.evaluation_pooling": "all_segment_mean",
            "features.normalization": "none", "features.unavailable_key_mode": "major",
            "model.architecture": "music2emo_single_task", "model.chord_sequence_length": 100,
            "model.root_vocabulary_size": 14, "model.quality_vocabulary_size": 14,
            "model.root_embedding_dimension": 4, "model.quality_embedding_dimension": 4,
            "model.transformer_layers": 2, "model.transformer_heads": 8, "model.transformer_feedforward_dimension": 64,
            "model.transformer_dropout": 0.1, "model.cls_token": "zeros", "model.padding_mask": "none",
            "model.hidden_dimensions": [512, 256], "model.activation": "relu", "model.output_labels": 56,
            "model.multitask": False, "model.knowledge_distillation": False,
            "optimization.optimizer": "adam", "optimization.loss": "author_frequency_weighted_bce",
            "optimization.betas": [0.9, 0.999], "optimization.epsilon": 1e-8, "optimization.weight_decay": 0.0,
            "optimization.early_stopping.monitor": "val_loss", "optimization.early_stopping.mode": "min",
            "optimization.scheduler": "none",
            "checkpoint.split": "validation", "checkpoint.minimize": "weighted_bce_loss", "checkpoint.ties": "earliest_epoch",
            "checkpoint.save_top_k": 5, "checkpoint.save_last": True,
            "evaluation.output_transform": "sigmoid", "evaluation.roc_auc": "macro", "evaluation.pr_auc": "macro_average_precision",
            "reproducibility.deterministic_algorithms": True, "reproducibility.sampling_seed": "sha256_per_seed_per_epoch",
            "runtime.matmul_precision": "medium", "runtime.attention_backend": "math"}
        if fusion_role:
            fixed.pop("features.source")
            fixed.update({"features.fusion": "concatenate" if role == "jamendo_mid_fusion" else "sum_logits_then_sigmoid",
                          "features.harmony_encoder": "shared" if role == "jamendo_mid_fusion" else "independent_per_branch",
                          "evaluation.accuracy": "labelwise",
                          "evaluation.f1": "macro", "evaluation.threshold_policy": "validation_macro_f1_global_grid",
                          "evaluation.threshold_grid.start": 0.0, "evaluation.threshold_grid.stop": 1.0,
                          "evaluation.threshold_grid.steps": 101})
            if role == "jamendo_mid_fusion":
                fixed["features.training_alignment"] = "shared_segment_range"
                expected_runs = JAMENDO_MID_CONDITIONS
            else:
                fixed["reproducibility.branch_seed"] = "sha256_position_paired"
                expected_runs = JAMENDO_LATE_CONDITIONS
            if runs != expected_runs:
                raise ConfigurationError(f"Use the fixed Jamendo {role.removeprefix('jamendo_')} conditions.")
            if run_name is not None and run_name not in runs:
                raise ConfigurationError(f"Unknown Jamendo fusion condition: {run_name}")
        if role == "jamendo_skey_training":
            fixed.update({"experiment_variant": "btc_chords_skey_audio_key",
                          "features.key_estimator": "skey_direct_audio", "features.unavailable_key_mode": "error",
                          "features.skey_commit": "80d9590e150b836a341dfc6dbf05aa20bc81d10b"})
        if role == "jamendo_sheetsage2_training":
            fixed.update({"experiment_variant": "sheetsage2_joint_chords_duration_weighted_key",
                          "features.chord_estimator": "sheetsage2_joint_paper",
                          "features.key_estimator": "sheetsage2_joint_paper",
                          "features.unavailable_key_mode": "error",
                          "features.chord_mapping": "sheetsage2_to_music2emo_v1_preserve_intervals",
                          "features.key_policy": "cumulative_duration_tonic_mode_ties_earliest",
                          "features.sheetsage2_revision": "b83c3d79a6d98bcffaf8f9ed112c90aa887887bc"})
        if role == "jamendo_chordformer_training":
            fixed.update({"experiment_variant": "chordformer_chords_cached_skey_audio_key",
                          "features.chord_estimator": "chordformer_five_model_submission",
                          "features.chord_mapping": "submission_to_music2emo_v1_preserve_intervals",
                          "features.key_estimator": "verified_existing_skey", "features.unavailable_key_mode": "error",
                          "features.chordformer_commit": "3da11c078c802b3f5a2ff0f6f8ec184a41537844",
                          "features.skey_commit": "80d9590e150b836a341dfc6dbf05aa20bc81d10b"})
        mappings = {"optimization.learning_rate": ("learning_rate", "float"), "optimization.batch_size": ("batch_size", "int"),
                    "optimization.epochs": ("epochs", "int"), "reproducibility.seeds": ("seeds", "int_list"),
                    "optimization.early_stopping.patience": ("early_stopping_patience", "int"),
                    "optimization.early_stopping.min_delta": ("early_stopping_min_delta", "float"),
                    "runtime.device": ("device", "str"), "runtime.num_workers": ("num_workers", "int"),
                    "execution.resume": ("resume", "bool"), "execution.overwrite": ("overwrite", "bool"),
                    "execution.validate_only": ("validate_only", "bool"), "execution.dry_run": ("dry_run", "bool")}
        if fusion_role:
            mappings["runtime.verify_checksums"] = ("verify_checksums", "bool")
            flat.setdefault("runtime.verify_checksums", True)
        if set(defaults) != _PATHS[role] or set(flat) != set(fixed) | set(mappings):
            raise ConfigurationError("Unknown or missing author-reproduction settings.")
        for name, expected in fixed.items():
            if flat[name] != expected or type(flat[name]) is not type(expected):
                raise ConfigurationError(f"{name} must be {expected!r}.")
        for name, (destination, kind) in mappings.items():
            defaults[destination] = _coerce(flat[name], kind, name)
        if fusion_role:
            defaults.update(runs=runs, run=run_name)
        return defaults, {"config_files": [{"path": name, "sha256": sha} for name, sha in files.items()],
                          "yaml_settings_before_cli_overrides": yaml_snapshot, "cli_overrides": {}}
    if role == "mert_segment_extraction":
        defaults = dict(data.pop("paths", {}))
        defaults.setdefault("model_path", None)
        flat = _flatten(data)
        if data.get("dataset") not in {"mtg_jamendo", "pmemo", "deam", "emomusic"}:
            raise ConfigurationError("Unsupported MERT extraction dataset.")
        fixed = {"dataset": data["dataset"], "extractor.frozen": True, "extractor.trust_remote_code": True,
                 "features.transformer_layers": 12, "features.dimension_per_layer": 768,
                 "features.exclude_embedding_state": True, "features.temporal_pooling": "mean",
                 "features.dtype": "float32", "audio.channels": "mono", "audio.tail_policy": "retain",
                 "output.format": "npy", "output.shape": [1, 12, 768]}
        settings = {"sources": ("source", "str_list"), "splits": ("splits", "str_list"),
                    "extractor.model_name": ("model_name", "str"), "extractor.revision": ("revision", "str"),
                    "audio.sample_rate_hz": ("sample_rate", "int"), "audio.segment_seconds": ("segment_seconds", "float"),
                    "audio.stem_alignment": ("stem_alignment", "str"),
                    "audio.max_alignment_seconds": ("max_alignment_seconds", "float"),
                    "audio.short_segment_policy": ("short_segment_policy", "str"),
                    "audio.minimum_input_samples": ("minimum_input_samples", "int"),
                    "runtime.device": ("device", "str"), "runtime.local_files_only": ("local_files_only", "bool"),
                    "execution.limit_per_split": ("limit_per_split", "optional_int"),
                    "execution.overwrite": ("overwrite", "bool"), "execution.fail_fast": ("fail_fast", "bool"),
                    "execution.dry_run": ("dry_run", "bool")}
        if set(defaults) != _PATHS[role] or set(flat) != set(fixed) | set(settings):
            raise ConfigurationError("Unknown or missing MERT segment extraction settings.")
        for name, expected in fixed.items():
            if flat[name] != expected or type(flat[name]) is not type(expected):
                raise ConfigurationError(f"{name} must be {expected!r}.")
        for name, (destination, kind) in settings.items():
            defaults[destination] = _coerce(flat[name], kind, name)
        return defaults, {"config_files": [{"path": name, "sha256": sha} for name, sha in files.items()],
                          "yaml_settings_before_cli_overrides": yaml_snapshot, "cli_overrides": {}}
    if role == "sheetsage2_extraction":
        defaults = dict(data.pop("paths", {}))
        flat = _flatten(data)
        fixed = {"dataset": "mtg_jamendo", "sources": ["mixture"],
                 "model.identifier": "m-a-p/SheetSage2", "model.revision": "b83c3d79a6d98bcffaf8f9ed112c90aa887887bc",
                 "model.backbone_identifier": "m-a-p/MERT-v2-FullSong",
                 "model.backbone_revision": "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42",
                 "inference.preset": "paper", "inference.prompts": ["timestamp", "downbeat_meter", "key", "chord_full"],
                 "harmony.chord_mapping": "sheetsage2_to_music2emo_v1_preserve_intervals",
                 "harmony.key_policy": "cumulative_duration_tonic_mode_ties_earliest",
                 "harmony.timeline_policy": "clip_audio_drop_zero_length_fill_chord_gaps_N",
                 "harmony.normalization": "C_major_A_minor"}
        settings = {"splits": ("splits", "str_list"), "runtime.device": ("device", "str"),
                    "runtime.local_files_only": ("local_files_only", "bool"),
                    "execution.limit_per_split": ("limit_per_split", "optional_int"),
                    "execution.fail_fast": ("fail_fast", "bool"), "execution.dry_run": ("dry_run", "bool")}
        if set(defaults) != _PATHS[role] or set(flat) != set(fixed) | set(settings):
            raise ConfigurationError("Unknown or missing SheetSage2 extraction settings.")
        for name, expected in fixed.items():
            if flat[name] != expected or type(flat[name]) is not type(expected):
                raise ConfigurationError(f"{name} must be {expected!r}.")
        for name, (destination, kind) in settings.items():
            defaults[destination] = _coerce(flat[name], kind, name)
        return defaults, {"config_files": [{"path": name, "sha256": sha} for name, sha in files.items()],
                          "yaml_settings_before_cli_overrides": yaml_snapshot, "cli_overrides": {}}
    if role == "chordformer_extraction":
        defaults = dict(data.pop("paths", {}))
        flat = _flatten(data)
        fixed = {"dataset": "mtg_jamendo", "sources": ["mixture"],
                 "chordformer_commit": "3da11c078c802b3f5a2ff0f6f8ec184a41537844",
                 "harmony.chord_vocabulary": "submission", "harmony.ensemble_models": 5,
                 "harmony.key_input": "verified_existing_skey",
                 "harmony.chord_mapping": "submission_to_music2emo_v1_preserve_intervals",
                 "harmony.normalization": "C_major_A_minor"}
        settings = {"splits": ("splits", "str_list"), "runtime.device": ("device", "str"),
                    "execution.limit_per_split": ("limit_per_split", "optional_int"),
                    "execution.fail_fast": ("fail_fast", "bool"), "execution.dry_run": ("dry_run", "bool")}
        if set(defaults) != _PATHS[role] or set(flat) != set(fixed) | set(settings):
            raise ConfigurationError("Unknown or missing ChordFormer extraction settings.")
        for name, expected in fixed.items():
            if flat[name] != expected or type(flat[name]) is not type(expected):
                raise ConfigurationError(f"{name} must be {expected!r}.")
        for name, (destination, kind) in settings.items():
            defaults[destination] = _coerce(flat[name], kind, name)
        return defaults, {"config_files": [{"path": name, "sha256": sha} for name, sha in files.items()],
                          "yaml_settings_before_cli_overrides": yaml_snapshot, "cli_overrides": {}}
    if role == "skey_extraction":
        defaults = dict(data.pop("paths", {}))
        defaults.setdefault("checkpoint", None)
        flat = _flatten(data)
        fixed = {"dataset": "mtg_jamendo", "skey_commit": "80d9590e150b836a341dfc6dbf05aa20bc81d10b",
                 "harmony.key_input": "whole_track_audio", "harmony.chord_input": "verified_existing_btc_predictions",
                 "harmony.normalization": "C_major_A_minor", "harmony.key_failure_policy": "error_no_fabricated_key"}
        settings = {"sources": ("source", "str_list"), "splits": ("splits", "str_list"),
                    "runtime.device": ("device", "str"), "execution.limit_per_split": ("limit_per_split", "optional_int"),
                    "execution.fail_fast": ("fail_fast", "bool"), "execution.dry_run": ("dry_run", "bool")}
        if set(defaults) != _PATHS[role] or set(flat) != set(fixed) | set(settings):
            raise ConfigurationError("Unknown or missing S-KEY extraction settings.")
        for name, expected in fixed.items():
            if flat[name] != expected or type(flat[name]) is not type(expected):
                raise ConfigurationError(f"{name} must be {expected!r}.")
        for name, (destination, kind) in settings.items():
            defaults[destination] = _coerce(flat[name], kind, name)
        return defaults, {"config_files": [{"path": name, "sha256": sha} for name, sha in files.items()],
                          "yaml_settings_before_cli_overrides": yaml_snapshot, "cli_overrides": {}}
    if role == "chord_key_extraction":
        defaults = dict(data.pop("paths", {}))
        flat = _flatten(data)
        if data.get("dataset") not in {"mtg_jamendo", "pmemo", "deam", "emomusic"}:
            raise ConfigurationError("Unsupported chord/key extraction dataset.")
        fixed = {"dataset": data["dataset"], "author.commit": "c5f7984f78d9928b841f25bd05a47377c12c58ca",
                 "harmony.chord_vocabulary": "large_170", "harmony.key_failure_policy": "record_none",
                 "harmony.normalization": "C_major_A_minor", "harmony.scope": "independent_per_source"}
        settings = {"sources": ("source", "str_list"), "splits": ("splits", "str_list"),
                    "harmony.midi_policy": ("midi_policy", "str"), "output.keep_midi": ("keep_midi", "bool"),
                    "execution.retry_failures": ("retry_failures", "bool"),
                    "runtime.device": ("device", "str"),
                    "execution.limit_per_split": ("limit_per_split", "optional_int"),
                    "execution.overwrite": ("overwrite", "bool"), "execution.fail_fast": ("fail_fast", "bool"),
                    "execution.dry_run": ("dry_run", "bool")}
        if set(defaults) != _PATHS[role] or set(flat) != set(fixed) | set(settings):
            raise ConfigurationError("Unknown or missing chord/key extraction settings.")
        for name, expected in fixed.items():
            if flat[name] != expected:
                raise ConfigurationError(f"{name} must be {expected!r}.")
        for name, (destination, kind) in settings.items():
            defaults[destination] = _coerce(flat[name], kind, name)
        return defaults, {"config_files": [{"path": name, "sha256": sha} for name, sha in files.items()],
                          "yaml_settings_before_cli_overrides": yaml_snapshot, "cli_overrides": {}}
    if role in {"pmemo_feature_manifest", "deam_feature_manifest", "emomusic_feature_manifest"}:
        defaults = dict(data.pop("paths", {}))
        flat = _flatten(data)
        settings = {"execution.overwrite": ("overwrite", "bool"),
                    "execution.dry_run": ("dry_run", "bool")}
        if (set(defaults) != _PATHS[role] or set(flat) != {"dataset", *settings}
                or flat.get("dataset") != role.removesuffix("_feature_manifest")):
            raise ConfigurationError("Unknown or missing emotion feature-manifest settings.")
        for name, (destination, kind) in settings.items():
            defaults[destination] = _coerce(flat[name], kind, name)
        return defaults, {"config_files": [{"path": name, "sha256": sha} for name, sha in files.items()],
                          "yaml_settings_before_cli_overrides": yaml_snapshot, "cli_overrides": {}}
    if role == "mtg_jamendo_feature_manifest":
        defaults = dict(data.pop("paths", {}))
        flat = _flatten(data)
        fixed = {"dataset": "mtg_jamendo"}
        settings = {"execution.overwrite": ("overwrite", "bool"),
                    "execution.dry_run": ("dry_run", "bool")}
        required = _PATHS[role] - {"audio_overrides"}
        if not required <= defaults.keys() or set(flat) != set(fixed) | set(settings):
            raise ConfigurationError("Unknown or missing Jamendo feature-manifest settings.")
        if flat["dataset"] != fixed["dataset"]:
            raise ConfigurationError("dataset must be mtg_jamendo.")
        for name, (destination, kind) in settings.items():
            defaults[destination] = _coerce(flat[name], kind, name)
        return defaults, {"config_files": [{"path": name, "sha256": sha} for name, sha in files.items()],
                          "yaml_settings_before_cli_overrides": yaml_snapshot, "cli_overrides": {}}
    if role in {"pmemo_separation_manifest", "deam_separation_manifest"}:
        defaults = dict(data.pop("paths", {}))
        flat = _flatten(data)
        settings = {"execution.all_tracks": ("all_tracks", "bool"),
                    "execution.overwrite": ("overwrite", "bool"),
                    "execution.dry_run": ("dry_run", "bool")}
        if not (_PATHS[role] - {"track_list"}) <= defaults.keys() or set(flat) != {"dataset", *settings}:
            raise ConfigurationError("Unknown or missing emotion separation-manifest settings.")
        dataset = role.removesuffix("_separation_manifest")
        if flat.get("dataset") != dataset:
            raise ConfigurationError(f"dataset must be {dataset}.")
        for name, (destination, kind) in settings.items():
            defaults[destination] = _coerce(flat[name], kind, name)
        return defaults, {"config_files": [{"path": name, "sha256": sha} for name, sha in files.items()],
                          "yaml_settings_before_cli_overrides": yaml_snapshot, "cli_overrides": {}}
    if role == "mtg_jamendo_separation_manifest":
        defaults = dict(data.pop("paths", {}))
        flat = _flatten(data)
        fixed = {"dataset": "mtg_jamendo"}
        settings = {"execution.all_tracks": ("all_tracks", "bool"),
                    "execution.overwrite": ("overwrite", "bool"),
                    "execution.dry_run": ("dry_run", "bool")}
        if not (_PATHS[role] - {"track_list"}) <= defaults.keys() or set(flat) != set(fixed) | set(settings):
            raise ConfigurationError("Unknown or missing MTG-Jamendo manifest settings.")
        for name, expected in fixed.items():
            if flat.get(name) != expected:
                raise ConfigurationError(f"{name} must be {expected!r}.")
        for name, (destination, kind) in settings.items():
            defaults[destination] = _coerce(flat[name], kind, name)
        return defaults, {"config_files": [{"path": name, "sha256": sha} for name, sha in files.items()],
                          "yaml_settings_before_cli_overrides": yaml_snapshot, "cli_overrides": {}}
    if role == "fma_manifest":
        defaults = dict(data.pop("paths", {}))
        excluded = data.pop("excluded_track_ids", [])
        if not isinstance(excluded, list) or any(not isinstance(value, str) or len(value) != 6 or not value.isdigit() for value in excluded):
            raise ConfigurationError("excluded_track_ids must contain six-digit FMA ID strings.")
        if len(set(excluded)) != len(excluded):
            raise ConfigurationError("excluded_track_ids must not contain duplicate IDs.")
        defaults["exclude_track_id"] = excluded
        fixed = {"dataset": "fma_medium"}
        settings = {"audio.mixture_extension": ("mixture_extension", "str"),
                    "execution.allow_missing": ("allow_missing", "bool"),
                    "execution.overwrite": ("overwrite", "bool"),
                    "execution.dry_run": ("dry_run", "bool")}
        flat = _flatten(data)
        if not {"tracks_csv", "output"} <= defaults.keys():
            raise ConfigurationError("FMA manifest configuration must provide paths.tracks_csv and paths.output.")
        if set(flat) != set(fixed) | set(settings):
            raise ConfigurationError("Unknown or missing FMA manifest configuration settings.")
        for name, expected in fixed.items():
            if flat.get(name) != expected:
                raise ConfigurationError(f"{name} must be {expected!r}.")
        for name, (destination, kind) in settings.items():
            defaults[destination] = _coerce(flat[name], kind, name)
        return defaults, {"config_files": [{"path": name, "sha256": sha} for name, sha in files.items()],
                          "yaml_settings_before_cli_overrides": yaml_snapshot, "cli_overrides": {}}
    if role == "ismir04_manifest":
        defaults = dict(data.pop("paths", {}))
        overrides = data.pop("filename_overrides", {})
        if not isinstance(overrides, dict) or any(not isinstance(value, str) or not value for value in overrides.values()):
            raise ConfigurationError("filename_overrides must map source paths to corrected relative paths.")
        defaults["filename_overrides"] = overrides
        flat = _flatten(data)
        fixed = {"dataset": "ismir04", "sources": _SOURCE_ORDER,
                 "duplicate_policy": "keep_training_then_validation_then_test"}
        settings = {"execution.audit_workers": ("audit_workers", "int"),
                    "execution.overwrite": ("overwrite", "bool"),
                    "execution.audit_only": ("audit_only", "bool")}
        if set(defaults) - {"filename_overrides"} != _PATHS[role] or set(flat) != set(fixed) | set(settings) | {"verified_legacy_stems"}:
            raise ConfigurationError("Unknown or missing ISMIR04 manifest configuration settings.")
        for name, expected in fixed.items():
            if flat.get(name) != expected:
                raise ConfigurationError(f"{name} must be {expected!r}.")
        verified = flat["verified_legacy_stems"]
        if not isinstance(verified, list) or any(not isinstance(item, str) or not item for item in verified):
            raise ConfigurationError("verified_legacy_stems must be a list of relative audio paths.")
        defaults["verified_legacy_stem"] = verified
        for name, (destination, kind) in settings.items():
            defaults[destination] = _coerce(flat[name], kind, name)
        return defaults, {"config_files": [{"path": name, "sha256": sha} for name, sha in files.items()],
                          "yaml_settings_before_cli_overrides": yaml_snapshot, "cli_overrides": {}}
    if role == "gtzan_manifest":
        defaults = dict(data.pop("paths", {}))
        excluded = data.pop("excluded_track_ids", [])
        flat = _flatten(data)
        flat.setdefault("execution.allow_missing", False)
        fixed = {"dataset": "gtzan", "sources": _SOURCE_ORDER}
        settings = {"audio.mixture_extension": ("mixture_extension", "str"),
                    "audio.stem_extension": ("stem_extension", "str"),
                    "execution.check_audio": ("check_audio", "bool"),
                    "execution.allow_missing": ("allow_missing", "bool"),
                    "execution.overwrite": ("overwrite", "bool"),
                    "execution.dry_run": ("dry_run", "bool")}
        if set(defaults) != _PATHS[role]:
            raise ConfigurationError("GTZAN manifest configuration must provide all split, audio, and output paths.")
        if set(flat) != set(fixed) | set(settings):
            raise ConfigurationError("Unknown or missing GTZAN manifest configuration settings.")
        for name, expected in fixed.items():
            if flat.get(name) != expected:
                raise ConfigurationError(f"{name} must be {expected!r}.")
        if not isinstance(excluded, list) or any(not isinstance(item, str) or not item for item in excluded):
            raise ConfigurationError("excluded_track_ids must be a list of track ID strings.")
        defaults["exclude_track_id"] = excluded
        for name, (destination, kind) in settings.items():
            defaults[destination] = _coerce(flat[name], kind, name)
        return defaults, {"config_files": [{"path": name, "sha256": sha} for name, sha in files.items()],
                          "yaml_settings_before_cli_overrides": yaml_snapshot, "cli_overrides": {}}
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
    dataset = metadata.get("dataset", "fma_medium")
    classes = {"fma_medium": 16, "gtzan": 10, "ismir04": 6}
    if role == "separation":
        classes["mtg_jamendo"] = 56
    if dataset not in classes and not (role == "separation" and dataset in {"pmemo", "deam"}):
        raise ConfigurationError(f"Unsupported dataset configuration: {dataset}")
    command_dataset = dataset.replace("_", "-")
    expected_command = {
        "extraction": f"extract {command_dataset}",
        "mid_fusion": f"train {command_dataset} mid-fusion",
        "late_fusion": f"train {command_dataset} late-fusion",
    }.get(role)
    if metadata.get("command", expected_command) != expected_command:
        raise ConfigurationError(f"Configuration command does not match {expected_command}.")
    if dataset in {"pmemo", "deam"} and "number_of_classes" in metadata:
        raise ConfigurationError("Emotion separation is label-independent; omit number_of_classes.")
    if dataset in classes and metadata.get("number_of_classes", classes[dataset]) != classes[dataset]:
        raise ConfigurationError(f"{dataset} configurations must describe {classes[dataset]} classes.")
    if role in {"mid_fusion", "late_fusion"}:
        fixed["task.number_of_classes"] = classes[dataset]
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
        if role == "extraction" and name == "audio.chunk_seconds" and name not in flat:
            defaults[destination] = None
            continue
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
    yaml_purposes: dict[str, tuple[str, str]] | None = None

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
        if self.yaml_purposes is not None:
            selector.add_argument("--purpose", choices=tuple(self.yaml_purposes), default=self.get_default("purpose"))
        if self.yaml_role in {"mid_fusion", "late_fusion", "jamendo_mid_fusion", "jamendo_late_fusion", "emotion_mid_fusion", "emotion_late_fusion", "deam_mid_fusion"}:
            selector.add_argument("--run")
        selected, _ = selector.parse_known_args(arguments)
        role, preset = self.yaml_role, self.yaml_default
        if self.yaml_purposes is not None:
            role, preset = self.yaml_purposes[selected.purpose]
        try:
            path = selected.config or default_config_path(preset)
            defaults, provenance = load_configuration(
                path, role, getattr(selected, "run", None)
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
    if role in {"mid_fusion", "late_fusion", "jamendo_mid_fusion", "jamendo_late_fusion", "emotion_mid_fusion", "emotion_late_fusion", "deam_mid_fusion"}:
        parser.add_argument("--run", help="Run one named condition; omit to run the complete YAML matrix.")
    if not isinstance(parser, YAMLArgumentParser):
        raise TypeError("Public YAML-configured tasks require YAMLArgumentParser.")
    parser.yaml_role = role
    parser.yaml_default = preset

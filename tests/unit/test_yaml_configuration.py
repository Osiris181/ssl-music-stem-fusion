from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from stemfusion.cli import build_parser
from stemfusion.configuration import default_config_path, load_configuration
from stemfusion.experiments.fma_medium.extraction import musicfm_config_from_args
from stemfusion.features._musicfm import extract_musicfm
from stemfusion.experiments.fma_medium.training import config_from_args as mid_config_from_args


class YAMLConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.parser = build_parser()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_yaml(self, data: dict, filename: str = "experiment.yaml") -> Path:
        path = self.root / filename
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    def extraction_arguments(self, path: Path, *extra: str) -> list[str]:
        return [
            "extract", "fma-medium", "--config", str(path),
            "--manifest", str(self.root / "manifest.jsonl"),
            "--musicfm-root", str(self.root / "musicfm"),
            "--output-dir", str(self.root / "features"),
            "--source", "guitar", "--dry-run", *extra,
        ]

    def extraction_profile(self, **overrides: object) -> dict:
        return {
            "feature_profile": str(default_config_path("features/musicfm_msd_layer7.yaml")),
            **overrides,
        }

    def assert_parse_error(self, arguments: list[str], message: str) -> None:
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            with self.assertRaises(SystemExit) as error:
                self.parser.parse_args(arguments)
        self.assertEqual(error.exception.code, 2)
        self.assertIn(message, output.getvalue())

    def test_yaml_controls_actual_extraction_configuration(self) -> None:
        path = self.write_yaml(self.extraction_profile(
            extractor={"checkpoint": "pretrained_fma.pt", "statistics": "fma_stats.json"},
            audio={"sample_rate_hz": 16000, "max_seconds": 5},
            features={"conformer_layer_index": 12, "dimension": 8},
            runtime={"device": "cpu", "flash_attention": True},
        ))
        args = self.parser.parse_args(self.extraction_arguments(path))
        config = musicfm_config_from_args(args)

        self.assertEqual(config.layer_index, 12)
        self.assertEqual(config.expected_dimension, 8)
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.max_seconds, 5.0)
        self.assertEqual(config.device, "cpu")
        self.assertTrue(config.is_flash)
        self.assertEqual(config.effective_checkpoint_path, self.root / "musicfm/data/pretrained_fma.pt")
        self.assertEqual(config.effective_stats_path.name, "fma_stats.json")

    def test_explicit_cli_overrides_scalars_booleans_and_split_list(self) -> None:
        path = self.write_yaml(self.extraction_profile(
            features={"conformer_layer_index": 12},
            runtime={"flash_attention": True},
        ))
        args = self.parser.parse_args(self.extraction_arguments(
            path, "--layer-index", "7", "--no-is-flash",
            "--split", "train", "--sample-rate", "24000",
        ))
        config = musicfm_config_from_args(args)

        self.assertEqual(config.layer_index, 7)
        self.assertFalse(config.is_flash)
        self.assertEqual(config.splits, ("train",))
        self.assertEqual(config.sample_rate, 24000)
        self.assertEqual(config.config_provenance["cli_overrides"]["layer_index"], 7)

    def test_relative_profiles_and_paths_can_supply_required_arguments(self) -> None:
        profile = self.root / "profile.yaml"
        profile.write_bytes(default_config_path("features/musicfm_msd_layer7.yaml").read_bytes())
        path = self.write_yaml({
            "feature_profile": "profile.yaml",
            "source": "piano",
            "paths": {
                "manifest": "inputs/manifest.jsonl",
                "musicfm_root": "encoder",
                "output_dir": "features",
            },
        })
        args = self.parser.parse_args(["extract", "fma-medium", "--config", str(path), "--dry-run"])
        config = musicfm_config_from_args(args)

        self.assertEqual(config.manifest_path, self.root / "inputs/manifest.jsonl")
        self.assertEqual(config.musicfm_root, self.root / "encoder")
        self.assertEqual(config.output_dir, self.root / "features")
        self.assertEqual(config.source, "piano")
        self.assertEqual(len(config.config_provenance["config_files"]), 2)

    def test_named_training_run_and_cli_seed_subset(self) -> None:
        experiment = default_config_path("experiments/fma_medium/mid_fusion.yaml")
        base = [
            "train", "fma-medium", "mid-fusion", "--config", str(experiment),
            "--run", "mixture_guitar_drums_piano",
            "--manifest", "manifest.jsonl", "--feature-dir", "features",
            "--output-dir", "output", "--dry-run",
        ]
        args = self.parser.parse_args(base)
        self.assertEqual(args.source, ["mixture", "guitar", "drums", "piano"])

        override = self.parser.parse_args([*base, "--seed", "32", "--seed", "64"])
        self.assertEqual(override.seeds, [32, 64])
        again = self.parser.parse_args(base)
        self.assertEqual(again.source, args.source)

    def test_training_yaml_defaults_and_cli_runtime_override(self) -> None:
        path = self.write_yaml({
            "training_profile": str(default_config_path("training/fma_medium_mid_fusion.yaml")),
            "optimization": {"batch_size": 4, "epochs": 3, "learning_rate": 0.002},
            "model": {"hidden_dimensions": [16, 8], "dropout": 0.1},
            "reproducibility": {"seeds": [123]},
            "sources": ["mixture"],
        })
        args = self.parser.parse_args([
            "train", "fma-medium", "mid-fusion", "--config", str(path),
            "--manifest", "manifest.jsonl", "--feature-dir", "features",
            "--output-dir", "output", "--device", "cpu", "--dry-run",
        ])
        config = mid_config_from_args(args)

        self.assertEqual(config.settings.batch_size, 4)
        self.assertEqual(config.settings.epochs, 3)
        self.assertEqual(config.settings.learning_rate, 0.002)
        self.assertEqual(config.settings.hidden_dimensions, (16, 8))
        self.assertEqual(config.seeds, (123,))
        self.assertEqual(config.settings.device, "cpu")
        self.assertEqual(config.provenance["cli_overrides"]["device"], "cpu")

    def test_custom_defaults_do_not_leak_to_a_later_parse(self) -> None:
        path = self.write_yaml(self.extraction_profile(features={"conformer_layer_index": 12}))
        self.parser.parse_args(self.extraction_arguments(path))
        defaults = self.parser.parse_args([
            "extract", "fma-medium", "--manifest", "manifest.jsonl",
            "--musicfm-root", "musicfm", "--output-dir", "features",
            "--source", "mixture", "--dry-run",
        ])
        self.assertEqual(defaults.layer_index, 7)
        self.assertFalse(defaults.is_flash)

    def test_unknown_yaml_setting_is_rejected(self) -> None:
        path = self.write_yaml(self.extraction_profile(features={"layer_indx": 7}))
        self.assert_parse_error(self.extraction_arguments(path), "features.layer_indx")

    def test_duplicate_yaml_keys_are_rejected(self) -> None:
        path = self.root / "duplicate.yaml"
        path.write_text("runtime:\n  device: cpu\n  device: cuda\n")
        self.assert_parse_error(self.extraction_arguments(path), "Duplicate YAML key")

    def test_missing_or_malformed_yaml_is_rejected(self) -> None:
        self.assert_parse_error(self.extraction_arguments(self.root / "missing.yaml"), "Cannot read YAML")
        path = self.root / "bad.yaml"
        path.write_text("features: [")
        self.assert_parse_error(self.extraction_arguments(path), "Cannot read YAML")
        path.write_text("[]")
        self.assert_parse_error(self.extraction_arguments(path), "YAML mapping")

    def test_unsafe_yaml_tags_are_rejected(self) -> None:
        path = self.root / "unsafe.yaml"
        path.write_text("!!python/tuple [1, 2]")
        self.assert_parse_error(self.extraction_arguments(path), "Cannot read YAML")

    def test_wrong_value_type_or_unsupported_architecture_is_rejected(self) -> None:
        path = self.write_yaml(self.extraction_profile(features={"conformer_layer_index": True}))
        self.assert_parse_error(self.extraction_arguments(path), "invalid value/type")
        path = self.write_yaml(self.extraction_profile(extractor={"architecture": "unimplemented"}))
        self.assert_parse_error(self.extraction_arguments(path), "not implemented")

    def test_cyclic_profile_and_wrong_task_are_rejected(self) -> None:
        first = self.write_yaml({"feature_profile": "second.yaml"}, "first.yaml")
        self.write_yaml({"feature_profile": "first.yaml"}, "second.yaml")
        self.assert_parse_error(self.extraction_arguments(first), "Cyclic")
        path = self.write_yaml(self.extraction_profile(command="train fma-medium mid-fusion"))
        self.assert_parse_error(self.extraction_arguments(path), "does not match")

    def test_unknown_training_run_is_rejected(self) -> None:
        self.assert_parse_error([
            "train", "fma-medium", "late-fusion", "--config",
            str(default_config_path("experiments/fma_medium/late_fusion.yaml")),
            "--run", "not_a_run",
        ], "Unknown/inapplicable run")

    def test_separator_defaults_are_loaded_from_yaml(self) -> None:
        path = self.write_yaml({
            "separator_profile": str(default_config_path("separators/htdemucs_6s.yaml")),
            "inference": {"device": "cpu", "shifts": 2, "overlap": 0.5, "jobs": 1},
            "paths": {"input_root": ["audio"], "output_root": "stems"},
        })
        args = self.parser.parse_args(["separate", "htdemucs", "--config", str(path), "--dry-run"])
        self.assertEqual(args.shifts, 2)
        self.assertEqual(args.overlap, 0.5)
        self.assertEqual(args.input_root, [self.root / "audio"])
        self.assertEqual(args.output_root, self.root / "stems")

    def test_resolved_extraction_report_records_values_and_file_hashes(self) -> None:
        audio = self.root / "audio.mp3"
        audio.write_bytes(b"synthetic")
        manifest = self.root / "manifest.jsonl"
        manifest.write_text(json.dumps({
            "track_id": "000002", "split": "train", "genre": "Rock",
            "label": 0, "paths": {"guitar": str(audio)},
        }) + "\n")
        model = self.root / "musicfm/data"
        model.mkdir(parents=True)
        (model / "pretrained_msd.pt").write_bytes(b"synthetic")
        (model / "msd_stats.json").write_text("{}")
        path = self.write_yaml(self.extraction_profile(
            features={"dimension": 3}, output={"files_per_source": ["train"]},
        ))
        args = self.parser.parse_args(self.extraction_arguments(path))
        config = musicfm_config_from_args(args)
        from dataclasses import replace
        summary = extract_musicfm(replace(config, dry_run=False), feature_extractor=lambda _: [1.0, 2.0, 3.0])
        report = json.loads((config.output_dir / "extraction_report_musicfm_guitar.json").read_text())
        resolved = report["resolved_config"]
        self.assertEqual(summary.succeeded, 1)
        self.assertEqual(resolved["expected_dimension"], 3)
        self.assertEqual(resolved["checkpoint_path"], str(model / "pretrained_msd.pt"))
        entry = resolved["config_provenance"]["config_files"][0]
        self.assertEqual(entry["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()

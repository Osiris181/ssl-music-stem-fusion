"""DEAM purpose-aware preparation and unchanged shared separation semantics."""

from __future__ import annotations

import ast
import contextlib
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import yaml

from stemfusion.cli import build_parser
from stemfusion.configuration import ConfigurationError, default_config_path, load_configuration
from stemfusion.data import deam_separation, pmemo_separation
from stemfusion.data.deam import build_feature_manifest, read_feature_manifest
from stemfusion.data.deam_separation import DEAMSeparationManifestConfig, build_separation_manifest
from stemfusion.experiments.deam import prepare_manifest, separation
from stemfusion.separation.htdemucs import HTDEMUCS_6S_STEMS, HTDemucsSeparationConfig, separate_manifest
from stemfusion.separation.manifest import read_manifest


class DEAMPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.splits = {}
        for split, track_id in (("train", "2"), ("validation", "3"), ("test", "4")):
            self.splits[split] = self.root / f"{split}.txt"
            self.splits[split].write_text(track_id + "\n")
        self.annotations = self.root / "annotations.csv"
        self.annotations.write_text("song_id,valence_mean,arousal_mean\n2,3.1,3.0\n3,3.5,3.3\n4,5.7,5.5\n")
        self.config = DEAMSeparationManifestConfig(self.splits, self.root / "audio",
            self.root / "stems", self.root / "separation.jsonl", all_tracks=True)

    def arguments(self, purpose):
        arguments = ["--purpose", purpose, "--mixture-root", str(self.config.mixture_root)]
        for split, path in self.splits.items():
            arguments.extend([f"--{split}-split", str(path)])
        if purpose == "features":
            arguments.extend(["--annotations", str(self.annotations),
                              "--htdemucs-root", str(self.config.output_root / "htdemucs_6s")])
        else:
            arguments.extend(["--output-root", str(self.config.output_root)])
        return arguments

    def invoke(self, arguments):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return prepare_manifest.main(arguments)

    def test_separation_preparation_policy_is_identical_to_pmemo_except_dataset_identity(self):
        source = inspect.getsource(deam_separation.build_separation_manifest).replace("DEAM", "PMEmo").replace("deam", "pmemo")
        candidate = ast.parse(source)
        reference = ast.parse(inspect.getsource(pmemo_separation.build_separation_manifest))
        for tree in (candidate, reference):
            tree.body[0].body.pop(0)  # Dataset-specific function documentation only.
        self.assertEqual(ast.dump(candidate), ast.dump(reference))

    def test_fixed_ids_splits_and_all_six_destinations_match_feature_manifest(self):
        report = build_separation_manifest(self.config)
        feature = self.root / "features.jsonl"
        build_feature_manifest(split_files=self.splits, annotations=self.annotations,
            mixture_root=self.config.mixture_root, htdemucs_root=self.config.output_root / "htdemucs_6s", output=feature)
        records = [json.loads(line) for line in self.config.output_path.read_text().splitlines()]
        for row, expected in zip(records, read_feature_manifest(feature), strict=True):
            self.assertEqual(row["dataset"], "deam")
            self.assertEqual(row["paths"], expected["paths"])
            self.assertEqual((row["track_id"], row["split"]), (expected["track_id"], expected["split"]))
            self.assertEqual(set(row["paths"]), {"mixture", *HTDEMUCS_6S_STEMS})
            self.assertNotIn("annotations", row)
        self.assertEqual(report["selected_split_counts"], {"train": 1, "validation": 1, "test": 1})
        self.assertFalse(report["binary_labels_created"])

    def test_metadata_only_dry_run_needs_no_audio_or_annotations_and_writes_nothing(self):
        self.annotations.unlink()
        report = build_separation_manifest(replace(self.config, dry_run=True))
        self.assertEqual(report["records"], 3)
        self.assertFalse(report["audio_files_checked"])
        self.assertFalse(self.config.output_path.exists())
        self.assertFalse(self.config.output_root.exists())
        self.assertFalse(self.config.mixture_root.exists())

    def test_overlap_malformed_ids_and_empty_splits_are_rejected(self):
        for value in ("2\n", "03\n", "../3\n", "track_3\n", ""):
            self.splits["validation"].write_text(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_separation_manifest(replace(self.config, dry_run=True))
        with self.assertRaisesRegex(ValueError, "Provide"):
            build_separation_manifest(replace(self.config, split_files={"train": self.splits["train"]}))

    def test_targeted_selection_preserves_order_and_cli_overrides_yaml_all_tracks(self):
        selection = self.root / "selection.txt"
        selection.write_text("4\n2\n")
        self.assertEqual(self.invoke(self.arguments("separation") + ["--output", str(self.config.output_path),
                                      "--track-list", str(selection)]), 0)
        records = read_manifest(self.config.output_path)
        self.assertEqual([(row.track_id, row.split) for row in records], [("4", "test"), ("2", "train")])
        report = json.loads(self.config.output_path.with_suffix(".report.json").read_text())
        self.assertFalse(report["all_tracks"])
        self.assertIsNotNone(report["track_list_sha256"])

    def test_invalid_selections_and_unselected_track_lists_follow_pmemo_policy(self):
        selection = self.root / "selection.txt"
        for text in ("", "2\n2\n", "99\n", "02\n", "../2\n"):
            selection.write_text(text)
            with self.assertRaises(ValueError):
                build_separation_manifest(replace(self.config, all_tracks=False, track_list=selection, dry_run=True))
        with self.assertRaisesRegex(ValueError, "track-list"):
            build_separation_manifest(replace(self.config, all_tracks=False, dry_run=True))
        selection.unlink()
        report = build_separation_manifest(replace(self.config, track_list=selection, dry_run=True))
        self.assertEqual(report["records"], 3)
        self.assertIsNone(report["track_list"])

    def test_manifests_reports_and_input_files_are_protected(self):
        build_separation_manifest(self.config)
        original = self.config.output_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            build_separation_manifest(self.config)
        self.config.output_path.unlink()
        with self.assertRaisesRegex(ValueError, "already exists"):
            build_separation_manifest(self.config)
        build_separation_manifest(replace(self.config, overwrite=True))
        self.assertEqual(self.config.output_path.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "must not replace"):
            build_separation_manifest(replace(self.config, output_path=self.splits["train"], overwrite=True))
        # Also protect an input whose name aliases the generated report path.
        report_input = self.root / "train.report.json"
        report_input.write_text("2\n")
        with self.assertRaisesRegex(ValueError, "must not replace"):
            build_separation_manifest(replace(self.config, split_files={**self.splits, "train": report_input},
                                              output_path=self.root / "train.jsonl", overwrite=True))
        self.assertEqual(self.splits["train"].read_text(), "2\n")

    def test_combined_feature_purpose_keeps_original_manifest_bytes(self):
        expected = self.root / "expected.jsonl"
        build_feature_manifest(split_files=self.splits, annotations=self.annotations,
            mixture_root=self.config.mixture_root, htdemucs_root=self.config.output_root / "htdemucs_6s", output=expected)
        actual = self.root / "actual.jsonl"
        self.assertEqual(self.invoke(self.arguments("features") + ["--output", str(actual)]), 0)
        self.assertEqual(actual.read_bytes(), expected.read_bytes())
        self.assertEqual(self.invoke(self.arguments("features") + ["--output", str(actual)]), 2)
        self.assertEqual(actual.read_bytes(), expected.read_bytes())

    def test_cli_aliases_purpose_switching_and_unsupported_options(self):
        parser = build_parser()
        for command, purpose in (("deam-manifest", "features"), ("deam-feature-manifest", "features"),
                                 ("deam-separation-manifest", "separation")):
            args = parser.parse_args(["data", command, "--dry-run"])
            self.assertEqual(args.purpose, purpose)
        args = parser.parse_args(["data", "deam-feature-manifest", "--purpose", "separation", "--dry-run"])
        self.assertEqual(args.purpose, "separation")
        self.assertEqual(args.output.name, "deam_separation.jsonl")
        for purpose, unsupported in (("features", ["--all-tracks"]),
                                     ("features", ["--output-root", str(self.root / "invalid")]),
                                     ("separation", ["--annotations", str(self.annotations)]),
                                     ("separation", ["--htdemucs-root", str(self.root / "invalid")])):
            self.assertEqual(self.invoke(self.arguments(purpose) + ["--dry-run", *unsupported]), 2)

    def test_yaml_paths_settings_and_inference_match_existing_datasets(self):
        values, _ = load_configuration(default_config_path("experiments/deam/separation.yaml"), "separation")
        for dataset in ("pmemo", "mtg_jamendo", "fma_medium"):
            reference, _ = load_configuration(default_config_path(f"experiments/{dataset}/separation.yaml"), "separation")
            for key in ("model_name", "device", "shifts", "overlap", "mp3_bitrate", "mp3_preset", "jobs", "batch_size"):
                self.assertEqual(values[key], reference[key], key)
        jobs, _ = load_configuration(default_config_path("experiments/deam/separation_manifest.yaml"), "deam_separation_manifest")
        feature, _ = load_configuration(default_config_path("experiments/deam/feature_manifest.yaml"), "deam_feature_manifest")
        self.assertEqual(jobs["mixture_root"], feature["mixture_root"])
        self.assertEqual(jobs["output_root"] / "htdemucs_6s", feature["htdemucs_root"])
        self.assertEqual(values["manifest"], jobs["output"])
        self.assertEqual(values["output_root"], jobs["output_root"])
        self.assertTrue(jobs["all_tracks"])

    def test_strict_yaml_rejects_wrong_dataset_types_and_training_metadata(self):
        for role, preset in (("deam_separation_manifest", "separation_manifest.yaml"), ("separation", "separation.yaml")):
            config = yaml.safe_load(default_config_path(f"experiments/deam/{preset}").read_text())
            path = self.root / "invalid.yaml"
            config["dataset"] = "unrelated"
            path.write_text(yaml.safe_dump(config))
            with self.assertRaises(ConfigurationError):
                load_configuration(path, role)
        preset = default_config_path("experiments/deam/separation_manifest.yaml")
        path.write_text(f"training_profile: {preset}\nexecution:\n  all_tracks: 'false'\n")
        with self.assertRaises(ConfigurationError):
            load_configuration(path, "deam_separation_manifest")
        preset = default_config_path("experiments/deam/separation.yaml")
        path.write_text(f"separator_profile: {preset}\nnumber_of_classes: 2\n")
        with self.assertRaisesRegex(ConfigurationError, "label-independent"):
            load_configuration(path, "separation")

    def test_complete_stems_are_reused_without_launching_demucs(self):
        build_separation_manifest(self.config)
        for row in read_manifest(self.config.output_path):
            for stem in HTDEMUCS_6S_STEMS:
                path = Path(row.paths[stem])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"existing stem")
        runner = Mock(side_effect=AssertionError("No separation needed"))
        config = HTDemucsSeparationConfig(input_roots=(), output_root=self.config.output_root)
        summary = separate_manifest(config, self.config.output_path, runner=runner)
        self.assertEqual((summary.skipped_existing, summary.succeeded, summary.failed), (3, 0, 0))
        runner.assert_not_called()

    def test_direct_commands_work_outside_checkout_and_dry_run_separation_is_readable(self):
        scripts = Path(prepare_manifest.__file__).parent
        self.assertFalse((scripts / "prepare_feature_manifest.py").exists())
        environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        result = subprocess.run([sys.executable, str(scripts / "prepare_manifest.py"),
            *self.arguments("separation"), "--output", str(self.config.output_path), "--dry-run"],
            cwd=self.root, env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["records"], 3)
        self.assertFalse(self.config.output_path.exists())
        build_separation_manifest(self.config)
        for row in read_manifest(self.config.output_path):
            path = Path(row.paths["mixture"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic audio")
        arguments = ["--manifest", str(self.config.output_path), "--output-root", str(self.config.output_root), "--dry-run"]
        result = subprocess.run([sys.executable, str(scripts / "separation.py"), *arguments],
            cwd=self.root, env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Pending separation: 3", result.stdout)
        self.assertIn("Check manifest and stems", result.stderr)
        self.assertFalse(self.config.output_root.exists())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(separation.main(arguments), 0)
        self.assertFalse(self.config.output_root.exists())


if __name__ == "__main__":
    unittest.main()

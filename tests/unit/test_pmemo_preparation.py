"""PMEmo preparation parity, protected manifests, and shared separation reuse."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from stemfusion.cli import build_parser
from stemfusion.configuration import ConfigurationError, default_config_path, load_configuration
from stemfusion.data.pmemo import build_feature_manifest
from stemfusion.data.pmemo_separation import PMEmoSeparationManifestConfig, build_separation_manifest
from stemfusion.experiments.pmemo import prepare_manifest, separation
from stemfusion.separation.htdemucs import HTDEMUCS_6S_STEMS, HTDemucsSeparationConfig, separate_manifest
from stemfusion.separation.manifest import read_manifest


class PMEmoPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.splits = {}
        for split, track_id in (("train", "1"), ("validation", "4"), ("test", "5")):
            self.splits[split] = self.root / f"{split}.txt"
            self.splits[split].write_text(track_id + "\n")
        self.annotations = self.root / "annotations.csv"
        self.annotations.write_text("musicId,Arousal(mean),Valence(mean)\n1,0.4,0.575\n4,0.25,0.3\n5,0.1,0.2\n")
        self.metadata = self.root / "metadata.csv"
        self.metadata.write_text("musicId,fileName,duration\n1,1.mp3,30.1\n4,4.mp3,31\n5,5.mp3,32\n")
        self.config = PMEmoSeparationManifestConfig(self.splits, self.root / "audio",
            self.root / "stems", self.root / "separation.jsonl", all_tracks=True)

    def tearDown(self):
        self.temp.cleanup()

    def arguments(self, purpose):
        arguments = ["--purpose", purpose, "--mixture-root", str(self.config.mixture_root)]
        for split, path in self.splits.items():
            arguments.extend([f"--{split}-split", str(path)])
        if purpose == "features":
            arguments.extend(["--annotations", str(self.annotations), "--metadata", str(self.metadata),
                              "--htdemucs-root", str(self.config.output_root / "htdemucs_6s")])
        else:
            arguments.extend(["--output-root", str(self.config.output_root)])
        return arguments

    def invoke(self, arguments):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return prepare_manifest.main(arguments)

    def test_separation_preserves_split_ids_and_feature_destinations(self):
        report = build_separation_manifest(self.config)
        self.assertEqual(report["selected_split_counts"], {"train": 1, "validation": 1, "test": 1})
        feature_path = self.root / "features.jsonl"
        build_feature_manifest(split_files=self.splits, annotations=self.annotations, metadata=self.metadata,
            mixture_root=self.config.mixture_root, htdemucs_root=self.config.output_root / "htdemucs_6s", output=feature_path)
        features = [json.loads(line) for line in feature_path.read_text().splitlines()]
        separated = [json.loads(line) for line in self.config.output_path.read_text().splitlines()]
        self.assertEqual([row["track_id"] for row in separated], ["1", "4", "5"])
        for feature, record in zip(features, separated):
            self.assertEqual(feature["split"], record["split"])
            self.assertEqual(feature["paths"], record["paths"])
            self.assertNotIn("annotations", record)
        self.assertFalse(report["binary_labels_created"])

    def test_metadata_only_dry_run_does_not_touch_audio_or_write(self):
        report = build_separation_manifest(replace(self.config, dry_run=True))
        self.assertEqual(report["records"], 3)
        self.assertFalse(report["audio_files_checked"])
        self.assertFalse(self.config.output_path.exists())
        self.assertFalse(self.config.output_root.exists())
        self.assertFalse(self.config.mixture_root.exists())

    def test_invalid_fixed_splits_are_rejected_without_exclusion(self):
        for value in ("1\n", "../4\n", "04\n", "track_0000004\n", ""):
            with self.subTest(value=value):
                self.splits["validation"].write_text(value)
                with self.assertRaises(ValueError):
                    build_separation_manifest(replace(self.config, dry_run=True))
        with self.assertRaisesRegex(ValueError, "Provide"):
            build_separation_manifest(replace(self.config, split_files={"train": self.splits["train"]}))

    def test_targeted_selection_preserves_requested_order_and_split(self):
        selection = self.root / "selection.txt"
        selection.write_text("5\n1\n")
        report = build_separation_manifest(replace(self.config, all_tracks=False, track_list=selection))
        self.assertEqual(report["records"], 2)
        self.assertEqual([(row.track_id, row.split) for row in read_manifest(self.config.output_path)],
                         [("5", "test"), ("1", "train")])
        self.assertIsNotNone(report["track_list_sha256"])

    def test_bad_selection_is_rejected_and_all_tracks_ignores_unused_list(self):
        selection = self.root / "selection.txt"
        for text in ("", "1\n1\n", "99\n", "../1\n"):
            selection.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                build_separation_manifest(replace(self.config, all_tracks=False, track_list=selection, dry_run=True))
        with self.assertRaisesRegex(ValueError, "track-list"):
            build_separation_manifest(replace(self.config, all_tracks=False, dry_run=True))
        selection.unlink()
        report = build_separation_manifest(replace(self.config, track_list=selection, dry_run=True))
        self.assertEqual(report["records"], 3)
        self.assertIsNone(report["track_list"])

    def test_existing_manifests_and_input_splits_are_protected(self):
        build_separation_manifest(self.config)
        original = self.config.output_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            build_separation_manifest(self.config)
        self.assertEqual(self.config.output_path.read_bytes(), original)
        self.config.output_path.unlink()
        with self.assertRaisesRegex(ValueError, "already exists"):
            build_separation_manifest(self.config)  # An existing report alone also protects output.
        build_separation_manifest(replace(self.config, overwrite=True))
        self.assertEqual(self.config.output_path.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "must not replace"):
            build_separation_manifest(replace(self.config, output_path=self.splits["train"], overwrite=True))
        self.assertEqual(self.splits["train"].read_text(), "1\n")

    def test_unified_feature_command_keeps_original_manifest_bytes(self):
        expected, output = self.root / "expected.jsonl", self.root / "features.jsonl"
        build_feature_manifest(split_files=self.splits, annotations=self.annotations, metadata=self.metadata,
            mixture_root=self.config.mixture_root, htdemucs_root=self.config.output_root / "htdemucs_6s", output=expected)
        arguments = self.arguments("features") + ["--output", str(output)]
        self.assertEqual(self.invoke(arguments + ["--dry-run"]), 0)
        self.assertFalse(output.exists())
        self.assertEqual(self.invoke(arguments), 0)
        self.assertEqual(output.read_bytes(), expected.read_bytes())
        self.assertEqual(self.invoke(arguments), 2)
        self.assertEqual(output.read_bytes(), expected.read_bytes())

    def test_unified_cli_aliases_and_purpose_defaults(self):
        parser = build_parser()
        for command, purpose in (("pmemo-manifest", "features"), ("pmemo-feature-manifest", "features"),
                                 ("pmemo-separation-manifest", "separation")):
            args = parser.parse_args(["data", command, "--dry-run"])
            self.assertEqual(args.purpose, purpose)
            self.assertEqual(args.command_handler, prepare_manifest.run_from_args)
        for purpose in ("separation", "features", "separation"):
            args = parser.parse_args(["data", "pmemo-manifest", "--purpose", purpose, "--dry-run"])
            self.assertEqual(args.output.name, f"pmemo_{'features' if purpose == 'features' else purpose}.jsonl")

    def test_cli_track_list_overrides_all_tracks_preset(self):
        selection = self.root / "selection.txt"
        selection.write_text("4\n")
        arguments = self.arguments("separation") + ["--track-list", str(selection), "--output", str(self.config.output_path)]
        self.assertEqual(self.invoke(arguments), 0)
        self.assertEqual([row.track_id for row in read_manifest(self.config.output_path)], ["4"])

    def test_cli_all_tracks_overrides_yaml_selection(self):
        preset = yaml.safe_load(default_config_path("experiments/pmemo/separation_manifest.yaml").read_text())
        preset["paths"]["track_list"] = str(self.root / "unused-missing-list.txt")
        preset["execution"]["all_tracks"] = False
        path = self.root / "selected.yaml"
        path.write_text(yaml.safe_dump(preset))
        arguments = self.arguments("separation") + ["--config", str(path), "--all-tracks", "--output", str(self.config.output_path)]
        self.assertEqual(self.invoke(arguments), 0)
        self.assertEqual(len(read_manifest(self.config.output_path)), 3)

    def test_wrong_purpose_options_and_dataset_yaml_are_rejected(self):
        for purpose, unused in (("features", ["--all-tracks"]), ("separation", ["--annotations", str(self.annotations)])):
            arguments = self.arguments(purpose) + unused + ["--output", str(self.config.output_path), "--dry-run"]
            self.assertEqual(self.invoke(arguments), 2)
        preset = yaml.safe_load(default_config_path("experiments/pmemo/separation_manifest.yaml").read_text())
        preset["dataset"] = "mtg_jamendo"
        path = self.root / "wrong.yaml"
        path.write_text(yaml.safe_dump(preset))
        with self.assertRaises(ConfigurationError):
            load_configuration(path, "pmemo_separation_manifest")

    def test_presets_match_jamendo_inference_and_existing_feature_paths(self):
        pmemo, _ = load_configuration(default_config_path("experiments/pmemo/separation.yaml"), "separation")
        jamendo, _ = load_configuration(default_config_path("experiments/mtg_jamendo/separation.yaml"), "separation")
        ignored = {"manifest", "output_root", "report"}
        self.assertEqual({k: v for k, v in pmemo.items() if k not in ignored},
                         {k: v for k, v in jamendo.items() if k not in ignored})
        features, _ = load_configuration(default_config_path("experiments/pmemo/feature_manifest.yaml"), "pmemo_feature_manifest")
        jobs, _ = load_configuration(default_config_path("experiments/pmemo/separation_manifest.yaml"), "pmemo_separation_manifest")
        self.assertEqual(features["htdemucs_root"], jobs["output_root"] / "htdemucs_6s")
        self.assertEqual(jobs["output"], pmemo["manifest"])
        for split in self.splits:
            self.assertEqual(features[f"{split}_split"], jobs[f"{split}_split"])
        self.assertEqual(features["mixture_root"], jobs["mixture_root"])
        for method, role in (("mert", "mert_segment_extraction"), ("chords_keys", "chord_key_extraction")):
            extraction, _ = load_configuration(default_config_path(f"experiments/pmemo/{method}.yaml"), role)
            self.assertEqual(features["output"], extraction["manifest"])

    def test_shared_backend_uses_exact_destinations_and_reuses_all_six_stems(self):
        build_separation_manifest(self.config)
        for record in read_manifest(self.config.output_path):
            path = Path(record.paths["mixture"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic mixture")
        calls = []
        def runner(command, **kwargs):
            calls.append(command)
            output = Path(command[command.index("--out") + 1])
            for stem in HTDEMUCS_6S_STEMS:
                alias = Path(command[-1])
                self.assertTrue(alias.is_symlink())
                path = output / "htdemucs_6s" / alias.stem / f"{stem}.mp3"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic stem")
            return subprocess.CompletedProcess(command, 0)
        config = HTDemucsSeparationConfig((), self.config.output_root, batch_size=1)
        report = separate_manifest(config, self.config.output_path, runner=runner)
        self.assertEqual((report.planned, report.succeeded, report.failed), (3, 3, 0))
        self.assertEqual(len(calls), 3)
        with patch("stemfusion.separation.htdemucs.tempfile.TemporaryDirectory", side_effect=AssertionError("Existing stems must be reused")):
            report = separate_manifest(config, self.config.output_path, runner=Mock(side_effect=AssertionError("Must not separate twice")))
        self.assertEqual(report.skipped_existing, 3)

    def test_direct_scripts_work_outside_repository_without_writing_during_dry_run(self):
        scripts = Path(prepare_manifest.__file__).parent
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        arguments = self.arguments("separation") + ["--output", str(self.config.output_path), "--dry-run"]
        result = subprocess.run([sys.executable, str(scripts / "prepare_manifest.py"), *arguments],
            cwd=self.root, env=env, text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout)["records"], 3)
        self.assertFalse(self.config.output_path.exists())
        build_separation_manifest(self.config)
        for record in read_manifest(self.config.output_path):
            path = Path(record.paths["mixture"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic mixture")
        arguments = ["--manifest", str(self.config.output_path), "--output-root", str(self.config.output_root), "--dry-run"]
        result = subprocess.run([sys.executable, str(scripts / "separation.py"), *arguments],
            cwd=self.root, env=env, text=True, capture_output=True, check=True)
        self.assertIn("Pending separation: 3", result.stdout)
        self.assertFalse(self.config.output_root.exists())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(separation.main(arguments), 0)
        self.assertFalse(self.config.output_root.exists())


if __name__ == "__main__":
    unittest.main()

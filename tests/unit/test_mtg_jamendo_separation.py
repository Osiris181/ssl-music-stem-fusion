from __future__ import annotations

import csv
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
from unittest.mock import patch

import numpy as np
import torch
import torchaudio
import soundfile as sf

from stemfusion.configuration import YAMLArgumentParser, default_config_path, load_configuration
from stemfusion.data.mtg_jamendo import JamendoSeparationManifestConfig, build_feature_manifest, build_separation_manifest, read_mood_splits
from stemfusion.experiments.mtg_jamendo import prepare_manifest
from stemfusion.separation.htdemucs import HTDemucsSeparationConfig, HTDEMUCS_6S_STEMS, separate_manifest
from stemfusion.separation.manifest import read_manifest
from stemfusion.experiments.mtg_jamendo.validate_separation import validate_audio


class JamendoSeparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.split_files = {}
        self.tags = [f"mood/theme---tag_{index}" for index in range(56)]
        for index, split in enumerate(("train", "validation", "test"), 1):
            p = self.root / f"{split}.tsv"
            with p.open("w", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "TAGS"])
                writer.writerow([f"track_{index:07d}", "artist", "album", f"{index:02d}/{index}.mp3", "60", *self.tags])
            self.split_files[split] = p
        self.selection = self.root / "tracks.txt"
        self.selection.write_text("track_0000001\ntrack_0000003\n")
        self.config = JamendoSeparationManifestConfig(self.split_files, self.root / "audio",
            self.root / "repairs", self.root / "manifest.jsonl", self.selection)

    def tearDown(self):
        self.temp.cleanup()

    def test_selected_tracks_preserve_tags_splits_and_numeric_destinations(self):
        report = build_separation_manifest(self.config)
        rows = [json.loads(line) for line in self.config.output_path.read_text().splitlines()]
        self.assertEqual([row["track_id"] for row in rows], ["track_0000001", "track_0000003"])
        self.assertEqual([row["split"] for row in rows], ["train", "test"])
        self.assertEqual(rows[0]["tags"], self.tags)
        self.assertNotIn("genre", rows[0])
        self.assertNotIn("label", rows[0])
        self.assertEqual(report["selected_split_counts"], {"train": 1, "validation": 0, "test": 1})
        self.assertEqual(Path(rows[0]["paths"]["guitar"]), self.config.output_root / "htdemucs_6s/1/guitar.mp3")
        self.assertEqual(len(read_manifest(self.config.output_path)), 2)
        self.assertFalse(self.config.output_root.exists())
        with self.assertRaisesRegex(ValueError, "already exists"):
            build_separation_manifest(self.config)

    def test_selection_and_split_errors_are_rejected(self):
        for text in ("track_9999999\n", "track_0000001\ntrack_0000001\n", ""):
            self.selection.write_text(text)
            with self.assertRaises(ValueError):
                build_separation_manifest(replace(self.config, dry_run=True))
        original = self.split_files["train"].read_text()
        self.split_files["test"].write_text(original)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_mood_splits(self.split_files)

    def test_all_tracks_and_dry_run_do_not_write(self):
        report = build_separation_manifest(replace(self.config, all_tracks=True, track_list=None, dry_run=True))
        self.assertEqual(report["records"], 3)
        self.assertFalse(self.config.output_path.exists())
        with self.assertRaisesRegex(ValueError, "track-list"):
            build_separation_manifest(replace(self.config, track_list=None, dry_run=True))

    def test_shared_backend_accepts_multilabel_metadata_and_reuses_outputs(self):
        build_separation_manifest(self.config)
        for record in read_manifest(self.config.output_path):
            p = Path(record.paths["mixture"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"synthetic audio")

        def runner(command, **kwargs):
            alias = Path(command[-1])
            self.assertTrue(alias.is_symlink())
            out = Path(command[command.index("--out") + 1])
            for stem in HTDEMUCS_6S_STEMS:
                p = out / "htdemucs_6s" / alias.stem / f"{stem}.mp3"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"synthetic stem")
            return subprocess.CompletedProcess(command, 0)

        config = HTDemucsSeparationConfig((), self.config.output_root, batch_size=1)
        summary = separate_manifest(config, self.config.output_path, runner=runner)
        self.assertEqual((summary.planned, summary.succeeded, summary.failed), (2, 2, 0))
        with patch("stemfusion.separation.htdemucs.tempfile.TemporaryDirectory", side_effect=AssertionError("Completed jobs must be reused")):
            reused = separate_manifest(config, self.config.output_path, runner=runner)
        self.assertEqual(reused.skipped_existing, 2)

    def test_settings_match_existing_datasets_and_direct_scripts_support_dry_runs(self):
        fma, _ = load_configuration(default_config_path("experiments/fma_medium/separation.yaml"), "separation")
        jamendo, _ = load_configuration(default_config_path("experiments/mtg_jamendo/separation.yaml"), "separation")
        ignored = {"manifest", "input_root", "output_root", "report"}
        self.assertEqual({k: v for k, v in fma.items() if k not in ignored}, {k: v for k, v in jamendo.items() if k not in ignored})
        repo = Path(__file__).resolve().parents[2]
        scripts = repo / "src/stemfusion/experiments/mtg_jamendo"
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        args = ["--train-split", str(self.split_files["train"]), "--validation-split", str(self.split_files["validation"]),
                "--test-split", str(self.split_files["test"]), "--track-list", str(self.selection),
                "--mixture-root", str(self.config.mixture_root), "--output-root", str(self.config.output_root),
                "--output", str(self.config.output_path), "--dry-run"]
        proc = subprocess.run([sys.executable, str(scripts / "prepare_manifest.py"), "--purpose", "separation", *args],
                              cwd=self.root, env=env, text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(proc.stdout)["records"], 2)
        self.assertFalse(self.config.output_path.exists())
        build_separation_manifest(self.config)
        for record in read_manifest(self.config.output_path):
            p = Path(record.paths["mixture"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"synthetic")
        proc = subprocess.run([sys.executable, str(scripts / "separation.py"), "--manifest", str(self.config.output_path),
                               "--output-root", str(self.config.output_root), "--dry-run"],
                              cwd=self.root, env=env, text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(proc.stdout)["planned"], 2)
        self.assertFalse(self.config.output_root.exists())

    def prepare_validation(self):
        build_separation_manifest(self.config)
        rows = [json.loads(line) for line in self.config.output_path.read_text().splitlines()]
        for row in rows:
            row["duration_seconds"] = 1.0
            for source in HTDEMUCS_6S_STEMS:
                path = Path(row["paths"][source])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic MP3")
        self.config.output_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return self.root / "validation.json", self.root / "validated.jsonl"

    def test_shared_feature_manifest_has_stage_specific_names_and_direct_command(self):
        from stemfusion.cli import build_parser

        feature, _ = load_configuration(default_config_path("experiments/mtg_jamendo/feature_manifest.yaml"),
                                        "mtg_jamendo_feature_manifest")
        separation, _ = load_configuration(default_config_path("experiments/mtg_jamendo/separation_manifest.yaml"),
                                           "mtg_jamendo_separation_manifest")
        mert, _ = load_configuration(default_config_path("experiments/mtg_jamendo/mert.yaml"),
                                     "mert_segment_extraction")
        harmony, _ = load_configuration(default_config_path("experiments/mtg_jamendo/chords_keys.yaml"),
                                        "chord_key_extraction")
        self.assertEqual(feature["output"].name, "mtg_jamendo_features.jsonl")
        self.assertEqual(feature["output"], mert["manifest"])
        self.assertEqual(feature["output"], harmony["manifest"])
        self.assertNotEqual(feature["output"], separation["output"])
        parser = build_parser()
        feature_args = parser.parse_args(["data", "mtg-jamendo-manifest", "--dry-run"])
        separation_args = parser.parse_args(["data", "mtg-jamendo-manifest", "--purpose", "separation", "--dry-run"])
        self.assertEqual(feature_args.command_handler, prepare_manifest.run_from_args)
        self.assertEqual(separation_args.command_handler, prepare_manifest.run_from_args)
        self.assertEqual(feature_args.purpose, "features")
        self.assertEqual(separation_args.purpose, "separation")
        self.assertEqual(feature_args.output, feature["output"])
        self.assertEqual(separation_args.output, separation["output"])
        for alias, purpose in (("mtg-jamendo-feature-manifest", "features"),
                               ("mtg-jamendo-separation-manifest", "separation")):
            args = parser.parse_args(["data", alias, "--dry-run"])
            self.assertEqual(args.purpose, purpose)
            self.assertEqual(args.command_handler, prepare_manifest.run_from_args)
        repo = Path(__file__).resolve().parents[2]
        scripts = repo / "src/stemfusion/experiments/mtg_jamendo"
        self.assertTrue((scripts / "prepare_manifest.py").is_file())
        self.assertFalse((scripts / "prepare_feature_manifest.py").exists())
        self.assertFalse((scripts / "prepare_separation_manifest.py").exists())
        self.assertFalse((scripts / "musicfm").exists())
        output = self.root / "features.jsonl"
        command = [sys.executable, str(scripts / "prepare_manifest.py"),
                   "--train-split", str(self.split_files["train"]),
                   "--validation-split", str(self.split_files["validation"]),
                   "--test-split", str(self.split_files["test"]),
                   "--mixture-root", str(self.config.mixture_root), "--htdemucs-root", str(self.config.output_root),
                   "--no-audio-overrides", "--output", str(output), "--dry-run"]
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        process = subprocess.run(command, cwd=self.root, env=env, text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(process.stdout)["records"], 3)
        self.assertFalse(output.exists())
        self.assertFalse(self.config.mixture_root.exists())

    def manifest_arguments(self, purpose):
        arguments = ["--purpose", purpose, "--mixture-root", str(self.config.mixture_root)]
        for split, path in self.split_files.items():
            arguments.extend([f"--{split}-split", str(path)])
        if purpose == "features":
            arguments.extend(["--htdemucs-root", str(self.config.output_root), "--no-audio-overrides"])
        else:
            arguments.extend(["--output-root", str(self.config.output_root), "--track-list", str(self.selection)])
        return arguments

    def test_unified_preparation_preserves_existing_builder_bytes_and_overwrite_guard(self):
        expected_features = self.root / "expected_features.jsonl"
        build_feature_manifest(split_files=self.split_files, mixture_root=self.config.mixture_root,
                               htdemucs_root=self.config.output_root, output=expected_features)
        build_separation_manifest(self.config)
        for purpose, expected in (("features", expected_features), ("separation", self.config.output_path)):
            with self.subTest(purpose=purpose), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                output = self.root / f"unified_{purpose}.jsonl"
                arguments = self.manifest_arguments(purpose) + ["--output", str(output)]
                self.assertEqual(prepare_manifest.main(arguments + ["--dry-run"]), 0)
                self.assertFalse(output.exists())
                self.assertEqual(prepare_manifest.main(arguments), 0)
                self.assertEqual(output.read_bytes(), expected.read_bytes())
                self.assertEqual(prepare_manifest.main(arguments), 2)
                self.assertEqual(output.read_bytes(), expected.read_bytes())
                self.assertEqual(prepare_manifest.main(arguments + ["--overwrite"]), 0)
                self.assertEqual(output.read_bytes(), expected.read_bytes())
        self.assertFalse(self.config.mixture_root.exists())
        self.assertFalse(self.config.output_root.exists())

    def test_unified_parser_does_not_leak_settings_between_purposes(self):
        parser = YAMLArgumentParser()
        prepare_manifest.configure_parser(parser)
        for purpose in ("features", "separation", "features", "separation"):
            args = parser.parse_args(["--purpose=" + purpose, "--dry-run"])
            role, preset = prepare_manifest._PRESETS[purpose]
            defaults, _ = load_configuration(default_config_path(preset), role)
            self.assertEqual(args.output, defaults["output"])
            self.assertEqual(args.purpose, purpose)
            self.assertEqual(args.yaml_provenance["config_files"][0]["path"], str(default_config_path(preset)))

    def test_unified_feature_preparation_preserves_validated_audio_overrides(self):
        build_separation_manifest(self.config)
        repaired = json.loads(self.config.output_path.read_text().splitlines()[0])
        repaired["paths"]["guitar"] = str(self.root / "validated_guitar.wav")
        overrides = self.root / "validated.jsonl"
        overrides.write_text(json.dumps(repaired) + "\n")
        expected = self.root / "expected_features.jsonl"
        build_feature_manifest(split_files=self.split_files, mixture_root=self.config.mixture_root,
                               htdemucs_root=self.config.output_root, output=expected, audio_overrides=overrides)
        arguments = self.manifest_arguments("features")
        arguments.remove("--no-audio-overrides")
        output = self.root / "actual_features.jsonl"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(prepare_manifest.main(arguments + ["--audio-overrides", str(overrides),
                                                                "--output", str(output)]), 0)
        self.assertEqual(output.read_bytes(), expected.read_bytes())
        self.assertEqual(json.loads(output.read_text().splitlines()[0])["paths"], repaired["paths"])

    def test_unified_installed_command_runs_both_purposes_with_cli_overrides(self):
        from stemfusion.cli import main

        for purpose in ("features", "separation"):
            output = self.root / f"installed_{purpose}.jsonl"
            arguments = self.manifest_arguments(purpose) + ["--output", str(output), "--dry-run"]
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(main(["data", "mtg-jamendo-manifest", *arguments]), 0)
            self.assertEqual(json.loads(stream.getvalue())["records"], 3 if purpose == "features" else 2)
            self.assertFalse(output.exists())

    def test_explicit_config_and_cli_paths_override_purpose_defaults(self):
        parser = YAMLArgumentParser()
        prepare_manifest.configure_parser(parser)
        for purpose, (role, preset) in prepare_manifest._PRESETS.items():
            output = self.root / f"custom_{purpose}.jsonl"
            args = parser.parse_args(["--purpose", purpose, "--config", str(default_config_path(preset)),
                                      "--output", str(output), "--dry-run"])
            self.assertEqual(args.output, output)
            self.assertEqual(args.yaml_provenance["cli_overrides"]["output"], str(output))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--purpose", "separation", "--config",
                               str(default_config_path(prepare_manifest._PRESETS["features"][1]))])

    def test_unified_preparation_rejects_wrong_purpose_options_and_invalid_purposes(self):
        for purpose, incompatible in (("features", ["--all-tracks"]),
                                      ("separation", ["--no-audio-overrides"])):
            with self.subTest(purpose=purpose), contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(prepare_manifest.main(["--purpose", purpose, *incompatible, "--dry-run"]), 2)
                self.assertIn("Options not supported", errors.getvalue())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare_manifest.main(["--purpose", "unknown", "--dry-run"])

    def test_unified_separation_all_tracks_overrides_configured_track_selection(self):
        output = self.root / "all_tracks.jsonl"
        arguments = self.manifest_arguments("separation")
        position = arguments.index("--track-list")
        del arguments[position:position + 2]
        arguments.extend(["--all-tracks", "--output", str(output)])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(prepare_manifest.main(arguments), 0)
        records = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(records), 3)
        self.assertEqual({row["split"] for row in records}, {"train", "validation", "test"})

    def test_audio_validation_keeps_silence_and_does_not_change_mp3_inputs(self):
        report_path, destination = self.prepare_validation()
        before = self.config.output_path.read_bytes()
        with patch("torchaudio.load", return_value=(torch.zeros(2, 8000), 8000)):
            report = validate_audio(self.config.output_path, self.config.output_root, report_path, destination)
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["audio_files"], 12)
        self.assertEqual(report["soundfile_fallbacks"], 0)
        self.assertEqual(destination.read_bytes(), before)
        self.assertFalse(list(self.config.output_root.rglob("*.wav")))

    def test_decoder_fallback_wav_preserves_samples_and_uses_separate_manifest(self):
        report_path, destination = self.prepare_validation()
        waveform = torch.full((2, 8000), 0.125, dtype=torch.float32)
        real_load, real_read = torchaudio.load, sf.read
        def loader(path):
            if str(path).endswith("guitar.mp3"):
                raise RuntimeError("Synthetic decoder incompatibility")
            if str(path).endswith(".mp3"):
                return waveform, 8000
            return real_load(path)
        def reader(path, **kwargs):
            if str(path).endswith("guitar.mp3"):
                return waveform.numpy().T, 8000
            return real_read(path, **kwargs)
        before = self.config.output_path.read_bytes()
        with patch("torchaudio.load", side_effect=loader), patch("soundfile.read", side_effect=reader):
            report = validate_audio(self.config.output_path, self.config.output_root, report_path,
                                    destination, write_wav_fallbacks=True)
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["soundfile_fallbacks"], 2)
        for row in [json.loads(line) for line in destination.read_text().splitlines()]:
            self.assertTrue(row["paths"]["guitar"].endswith(".wav"))
            self.assertEqual(sf.info(row["paths"]["guitar"]).subtype, "FLOAT")
            np.testing.assert_array_equal(sf.read(row["paths"]["guitar"], dtype="float32")[0], waveform.numpy().T)
        self.assertEqual(self.config.output_path.read_bytes(), before)

    def test_invalid_duration_prevents_publication_of_downstream_manifest(self):
        report_path, destination = self.prepare_validation()
        with patch("torchaudio.load", return_value=(torch.zeros(2, 16000), 8000)):
            report = validate_audio(self.config.output_path, self.config.output_root, report_path, destination)
        self.assertEqual(report["failed"], 12)
        self.assertIsNone(report["validated_manifest"])
        self.assertFalse(destination.exists())
        self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()

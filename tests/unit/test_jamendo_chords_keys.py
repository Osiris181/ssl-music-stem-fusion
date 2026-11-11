from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from stemfusion.configuration import ConfigurationError, default_config_path, load_configuration
from stemfusion.features.chords_keys import (
    LEGACY_IMPLEMENTATION_SHA256, NORMAL_PIPELINE_REVISION, _supports_cqt_retry,
    ChordKeyExtractionConfig, extract_chords_keys, normalize_chords, parse_intervals,
    pitch_events, prediction_intervals,
)


class ParameterError(ValueError):
    """Synthetic librosa-style failure; unit fixtures need no audio decoder."""


class ChordKeyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.audio = self.root / "fixture.wav"
        self.audio.write_bytes(b"synthetic audio")
        self.manifest = self.root / "manifest.jsonl"
        self.row = {"track_id": "track_0000001", "relative_path": "01/1.mp3", "split": "train",
                    "tags": ["mood/theme---happy"], "paths": {"mixture": str(self.audio), "drums": str(self.audio)}}
        self.manifest.write_text(json.dumps(self.row) + "\n")
        self.config = ChordKeyExtractionConfig(self.manifest, self.root / "features", self.root / "author",
            self.root / "btc.yaml", self.root / "btc.pt", self.root / "report.json",
            sources=("mixture", "drums"), splits=("train",), device="cpu")
        self.settings_patch = patch("stemfusion.features.chords_keys._settings", return_value={"fixture": True})
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def runtime(audio, output):
        (output / "chords.lab").write_text("0.000 1.000 C\n1.000 2.000 G\n")
        (output / "normalized_chords.lab").write_text("0.000 1.000 C\n1.000 2.000 G\n")
        (output / "key.lab").write_text("C major\n")
        (output / "chords.mid").write_bytes(b"synthetic MIDI")
        return {"chord_intervals": 2, "midi_notes": 5, "key": "C major", "key_status": "estimated", "key_error": None}

    def test_released_interval_loop_matches_independent_reference(self):
        rng = np.random.default_rng(32)
        vocab = {0: "C", 1: "G", 2: "N"}
        for frames in (1, 2, 107, 108, 109, 215, 216, 217, 324):
            timestep = 108
            padding = timestep - frames % timestep
            pred = rng.integers(0, 3, size=frames + padding)
            expected, start, previous = [], 0.0, None
            for block in range(len(pred) // timestep):
                for i in range(timestep):
                    value = int(pred[block * timestep + i])
                    if block == 0 and i == 0:
                        previous = value
                        continue
                    boundary = (10 / 108) * (block * timestep + i)
                    if value != previous:
                        expected.append('%.3f %.3f %s\n' % (start, boundary, vocab[previous]))
                        start, previous = boundary, value
                    if block == len(pred) // timestep - 1 and i + padding == timestep:
                        if start != boundary:
                            expected.append('%.3f %.3f %s\n' % (start, boundary, vocab[previous]))
                        break
            self.assertEqual(prediction_intervals(pred, frames, timestep, 10 / 108, vocab), expected)

    def test_normalization_matches_major_minor_flat_and_no_key_rules(self):
        lines = ["0.000 1.000 D:min\n", "1.000 2.000 A\n", "2.000 3.000 N\n", "3.000 4.000 X\n"]
        major = normalize_chords(lines, "D major")
        self.assertEqual(major[:2], ["0.000 1.000 C:min\n", "1.000 2.000 G\n"])
        self.assertEqual(major[2:], lines[2:])
        self.assertEqual(normalize_chords([lines[0]], "B minor"), ["0.000 1.000 C:min\n"])
        self.assertEqual(normalize_chords(["0.000 1.000 C#\n"], "D- major"), ["0.000 1.000 C\n"])
        self.assertEqual(normalize_chords(lines, "None"), lines)
        with self.assertRaises(ValueError):
            normalize_chords(lines, "None major")

    def test_single_interval_midi_edge_is_explicit_not_silently_changed(self):
        intervals = [(0.0, 2.0, "C")]
        bitmaps = [[1 if pitch in (0, 4, 7) else 0 for pitch in range(12)]]
        self.assertEqual(pitch_events(intervals, bitmaps), [])
        self.assertEqual(pitch_events(intervals, bitmaps, "complete_intervals"),
                         [(0.0, 2.0, 48), (0.0, 2.0, 52), (0.0, 2.0, 55)])

    def test_pitch_event_transitions_merge_shared_pitches(self):
        intervals = [(0.0, 1.0, "C"), (1.0, 2.0, "G"), (2.0, 3.0, "N")]
        bitmaps = [[int(p in chord) for p in range(12)] for chord in ((0, 4, 7), (2, 7, 11), ())]
        self.assertEqual(sorted(pitch_events(intervals, bitmaps)),
                         sorted([(0, 1, 48), (0, 1, 52), (0, 2, 55), (1, 2, 50), (1, 2, 59)]))

    def test_interval_validation_rejects_bad_timings_and_empty_results(self):
        for lines in ([], ["0 0 C"], ["1 2 C"], ["0 nan C"], ["0 1 C", "2 3 G"]):
            with self.assertRaises(ValueError):
                parse_intervals(lines)

    def test_separate_sources_lazy_initialization_and_verified_resume(self):
        runtime = Mock(side_effect=self.runtime)
        factory = Mock(return_value=runtime)
        report = extract_chords_keys(self.config, runtime_factory=factory)
        self.assertEqual((report["extracted"], report["failed"], report["exit_code"]), (2, 0, 0))
        factory.assert_called_once_with(self.config)
        self.assertEqual(runtime.call_count, 2)
        for source in self.config.sources:
            folder = self.config.output_dir / source / "01/1"
            metadata = json.loads((folder / "metadata.json").read_text())
            self.assertEqual(metadata["source"], source)
            self.assertEqual(metadata["audio"]["path"], str(self.audio))
        no_runtime = Mock(side_effect=AssertionError("Cache reuse must not initialize BTC"))
        reused = extract_chords_keys(self.config, runtime_factory=no_runtime)
        self.assertEqual(reused["reused"], 2)
        no_runtime.assert_not_called()

    def test_changed_audio_is_not_reused(self):
        extract_chords_keys(self.config, runtime_factory=Mock(return_value=self.runtime))
        self.audio.write_bytes(b"changed fixture")
        report = extract_chords_keys(self.config, runtime_factory=Mock(side_effect=AssertionError()))
        self.assertEqual(report["failed"], 2)

    def test_changed_settings_are_not_reused(self):
        extract_chords_keys(self.config, runtime_factory=Mock(return_value=self.runtime))
        with patch("stemfusion.features.chords_keys._settings", return_value={"fixture": False}):
            report = extract_chords_keys(self.config, runtime_factory=Mock(side_effect=AssertionError()))
        self.assertEqual(report["failed"], 2)

    def test_corrupt_output_is_not_reused(self):
        extract_chords_keys(self.config, runtime_factory=Mock(return_value=self.runtime))
        (self.config.output_dir / "mixture/01/1/chords.lab").write_text("corrupted")
        report = extract_chords_keys(self.config, runtime_factory=Mock(side_effect=AssertionError()))
        self.assertEqual((report["failed"], report["reused"]), (1, 1))

    def test_failed_overwrite_preserves_previous_complete_output(self):
        extract_chords_keys(self.config, runtime_factory=Mock(return_value=self.runtime))
        folder = self.config.output_dir / "mixture/01/1"
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        broken = Mock(side_effect=ValueError("synthetic failure"))
        report = extract_chords_keys(replace(self.config, overwrite=True, sources=("mixture",)),
                                    runtime_factory=Mock(return_value=broken))
        self.assertEqual(report["failed"], 1)
        self.assertEqual({p.name: p.read_bytes() for p in folder.iterdir()}, before)
        self.assertEqual(list(folder.parent.glob(".1.*")), [])

    def test_failed_promotion_restores_the_previous_cache(self):
        extract_chords_keys(self.config, runtime_factory=Mock(return_value=self.runtime))
        folder = self.config.output_dir / "mixture/01/1"
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        original_replace = Path.replace

        def fail_promotion(path, destination):
            if path.name.startswith('.1.') and '.previous.' not in path.name and Path(destination) == folder:
                raise OSError("synthetic promotion failure")
            return original_replace(path, destination)

        with patch.object(Path, "replace", fail_promotion):
            report = extract_chords_keys(replace(self.config, sources=("mixture",), overwrite=True),
                                        runtime_factory=Mock(return_value=self.runtime))
        self.assertEqual(report["failed"], 1)
        self.assertEqual({p.name: p.read_bytes() for p in folder.iterdir()}, before)

    def test_failed_restoration_keeps_a_recoverable_previous_cache(self):
        extract_chords_keys(self.config, runtime_factory=Mock(return_value=self.runtime))
        folder = self.config.output_dir / "mixture/01/1"
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        original_replace = Path.replace

        def fail_promote_and_restore(path, destination):
            if path.name.startswith('.1.') and Path(destination) == folder:
                raise OSError("synthetic promotion/restoration failure")
            return original_replace(path, destination)

        with patch.object(Path, "replace", fail_promote_and_restore):
            report = extract_chords_keys(replace(self.config, sources=("mixture",), overwrite=True),
                                        runtime_factory=Mock(return_value=self.runtime))
        self.assertEqual(report["failed"], 1)
        backups = list(folder.parent.glob(".1.previous.*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual({p.name: p.read_bytes() for p in backups[0].iterdir()}, before)

    def test_input_mutation_cannot_publish_a_complete_cache(self):
        def mutate(audio, output):
            result = self.runtime(audio, output)
            audio.write_bytes(b"mutated while processing")
            return result
        config = replace(self.config, sources=("mixture",))
        report = extract_chords_keys(config, runtime_factory=Mock(return_value=mutate))
        self.assertEqual(report["failed"], 1)
        self.assertFalse((config.output_dir / "mixture/01/1").exists())

    def test_missing_key_is_reported_not_dropped_or_changed_to_major(self):
        def unavailable(audio, output):
            result = self.runtime(audio, output)
            (output / "key.lab").write_text("None\n")
            return {**result, "key": "None", "key_status": "unavailable", "key_error": "empty MIDI"}
        report = extract_chords_keys(self.config, runtime_factory=Mock(return_value=unavailable))
        self.assertEqual((report["extracted"], report["failed"], report["key_unavailable"]), (2, 0, 2))
        self.assertEqual((self.config.output_dir / "drums/01/1/key.lab").read_text(), "None\n")
        self.assertEqual(report["retry_attempted"], 0)

    def retry_runtime(self, *, retry_error=None):
        fixture = self.runtime

        class Runtime:
            def __init__(self):
                self.calls = []

            def __call__(self, audio, output):
                self.calls.append("normal")
                raise ParameterError("Input signal length=1 is too short for 6-octave CQT")

            def retry_with_cqt_padding(self, audio, output):
                self.calls.append("retry")
                if retry_error is not None:
                    raise retry_error
                return {**fixture(audio, output), "cqt_repair": {"padded_samples": 1}}

        return Runtime()

    def test_default_two_pass_retry_saves_initial_and_final_reports_then_reuses(self):
        runtime = self.retry_runtime()
        report = extract_chords_keys(self.config, runtime_factory=lambda _: runtime)
        self.assertEqual(runtime.calls, ["normal", "normal", "retry", "retry"])
        self.assertEqual((report["initial_failed"], report["retry_attempted"], report["repaired"], report["failed"]), (2, 2, 2, 0))
        self.assertEqual(report["exit_code"], 0)
        initial = json.loads(Path(report["initial_report"]).read_text())
        self.assertEqual((initial["failed"], initial["repaired"]), (2, 0))
        self.assertTrue(all(row["status"] == "failed" for row in initial["records"]))
        final = json.loads(self.config.report.read_text())
        self.assertTrue(all(row["status"] == "repaired" and row["attempts"] == 2 for row in final["records"]))
        self.assertTrue(all("initial_error" in row and "error" not in row for row in final["records"]))
        no_model = Mock(side_effect=AssertionError("Completed repairs must reuse without loading BTC"))
        reused = extract_chords_keys(replace(self.config, retry_failures=False), runtime_factory=no_model)
        self.assertEqual((reused["reused"], reused["failed"], reused["retry_attempted"]), (2, 0, 0))
        no_model.assert_not_called()

    def test_retry_can_be_disabled_and_fail_fast_does_not_enter_retry_pass(self):
        for changes in ({"retry_failures": False}, {"fail_fast": True}):
            with self.subTest(changes=changes):
                runtime = self.retry_runtime()
                report = extract_chords_keys(replace(self.config, **changes), runtime_factory=lambda _: runtime)
                self.assertNotIn("retry", runtime.calls)
                self.assertEqual((report["repaired"], report["retry_attempted"], report["exit_code"]), (0, 0, 1))
                self.assertIsNone(report["initial_report"])

    def test_failed_retry_runs_once_and_retains_both_errors(self):
        runtime = self.retry_runtime(retry_error=RuntimeError("synthetic retry failure"))
        report = extract_chords_keys(replace(self.config, sources=("mixture",)), runtime_factory=lambda _: runtime)
        self.assertEqual(runtime.calls, ["normal", "retry"])
        self.assertEqual((report["failed"], report["retry_attempted"], report["repaired"], report["exit_code"]), (1, 1, 0, 1))
        record = report["records"][0]
        self.assertIn("too short", record["initial_error"])
        self.assertIn("synthetic retry failure", record["retry_error"])
        self.assertFalse((self.config.output_dir / "mixture/01/1").exists())

    def test_retry_classifier_rejects_unrelated_and_unsupported_failures(self):
        for error in (RuntimeError("Input signal length=1 is too short for 6-octave CQT"),
                      ParameterError("Input signal length=0 is too short for 6-octave CQT"),
                      ParameterError("Input signal length=2 is too short for 6-octave CQT"),
                      ParameterError("Input signal length=1 is too short for 7-octave CQT"),
                      FileNotFoundError("missing audio")):
            self.assertFalse(_supports_cqt_retry(error))
        self.assertTrue(_supports_cqt_retry(ParameterError("Input signal length=1 is too short for 6-octave CQT")))

    @staticmethod
    def legacy_settings():
        return {"schema": 1, "implementation_sha256": LEGACY_IMPLEMENTATION_SHA256,
                "checkpoint_sha256": "fixture-checkpoint", "device": "cpu", "midi_policy": "music2emotion_released",
                "package_versions": {"librosa": "fixture-version"}}

    @staticmethod
    def current_settings(legacy):
        return {**legacy, "schema": 2, "implementation_sha256": "fixture-current-implementation",
                "normal_pipeline_revision": NORMAL_PIPELINE_REVISION,
                "cqt_retry": {"policy": "right_zero_pad_to_minimum", "minimum_input_samples": 2}}

    def test_verified_legacy_outputs_stay_byte_identical_while_missing_view_is_repaired(self):
        legacy = self.legacy_settings()
        with patch("stemfusion.features.chords_keys._settings", return_value=legacy):
            extract_chords_keys(replace(self.config, sources=("mixture",)), runtime_factory=lambda _: self.runtime)
        folder = self.config.output_dir / "mixture/01/1"
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        runtime = self.retry_runtime()
        with patch("stemfusion.features.chords_keys._settings", return_value=self.current_settings(legacy)):
            report = extract_chords_keys(self.config, runtime_factory=lambda _: runtime)
            self.assertEqual((report["legacy_reused"], report["reused"], report["repaired"], report["failed"]), (1, 1, 1, 0))
            no_model = Mock(side_effect=AssertionError("Mixed legacy/current caches must reuse"))
            again = extract_chords_keys(self.config, runtime_factory=no_model)
            self.assertEqual((again["reused"], again["failed"]), (2, 0))
            no_model.assert_not_called()
        self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})
        self.assertEqual(runtime.calls, ["normal", "retry"])

    def test_legacy_compatibility_does_not_ignore_scientific_or_package_changes(self):
        legacy = self.legacy_settings()
        config = replace(self.config, sources=("mixture",))
        with patch("stemfusion.features.chords_keys._settings", return_value=legacy):
            extract_chords_keys(config, runtime_factory=lambda _: self.runtime)
        current = self.current_settings(legacy)
        for key, changed in (("checkpoint_sha256", "other-weights"), ("device", "cuda"),
                             ("midi_policy", "complete_intervals"), ("package_versions", {"librosa": "other-version"}),
                             ("normal_pipeline_revision", "unknown-pipeline")):
            with self.subTest(key=key), patch("stemfusion.features.chords_keys._settings", return_value={**current, key: changed}):
                factory = Mock(side_effect=AssertionError("Incompatible caches must not be overwritten/repaired"))
                report = extract_chords_keys(config, runtime_factory=factory)
                self.assertEqual((report["failed"], report["retry_attempted"], report["reused"]), (1, 0, 0))
                factory.assert_not_called()

    def test_unknown_legacy_implementation_and_corrupted_legacy_files_are_rejected(self):
        legacy = self.legacy_settings()
        config = replace(self.config, sources=("mixture",))
        with patch("stemfusion.features.chords_keys._settings", return_value={**legacy, "implementation_sha256": "unknown-old-code"}):
            extract_chords_keys(config, runtime_factory=lambda _: self.runtime)
        current = self.current_settings(legacy)
        with patch("stemfusion.features.chords_keys._settings", return_value=current):
            self.assertEqual(extract_chords_keys(config, runtime_factory=lambda _: self.runtime)["failed"], 1)
        with patch("stemfusion.features.chords_keys._settings", return_value=legacy):
            extract_chords_keys(replace(config, overwrite=True), runtime_factory=lambda _: self.runtime)
        (config.output_dir / "mixture/01/1/chords.mid").write_bytes(b"corrupted legacy MIDI")
        with patch("stemfusion.features.chords_keys._settings", return_value=current):
            report = extract_chords_keys(config, runtime_factory=lambda _: self.runtime)
            self.assertEqual((report["failed"], report["retry_attempted"]), (1, 0))

    def test_dry_run_does_not_read_audio_or_dependencies_or_write_outputs(self):
        self.audio.unlink()
        factory = Mock(side_effect=AssertionError())
        report = extract_chords_keys(replace(self.config, dry_run=True), runtime_factory=factory)
        self.assertEqual(report["planned"], 2)
        self.assertFalse(self.config.output_dir.exists())
        self.assertFalse(self.config.report.exists())
        factory.assert_not_called()

    def test_fail_fast_counts_unprocessed_sources(self):
        report = extract_chords_keys(replace(self.config, fail_fast=True),
            runtime_factory=Mock(return_value=Mock(side_effect=ValueError("synthetic failure"))))
        self.assertEqual((report["failed"], report["unprocessed"], len(report["records"])), (1, 1, 1))

    def test_config_validation_rejects_unknown_sources_and_negative_limits(self):
        for config in (replace(self.config, sources=("other",)), replace(self.config, limit_per_split=0),
                       replace(self.config, midi_policy="unknown"), replace(self.config, splits=("holdout",))):
            with self.assertRaises(ValueError):
                extract_chords_keys(config)

    def test_yaml_defaults_paths_and_strict_unknown_setting_validation(self):
        preset = default_config_path("experiments/mtg_jamendo/chords_keys.yaml")
        values, _ = load_configuration(preset, "chord_key_extraction")
        self.assertEqual(values["source"], ["mixture", "guitar", "drums", "piano", "bass", "vocals"])
        self.assertEqual(values["midi_policy"], "music2emotion_released")
        self.assertTrue(values["keep_midi"])
        self.assertTrue(values["retry_failures"])
        self.assertEqual(values["manifest"].name, "mtg_jamendo_features.jsonl")
        repo = Path(__file__).resolve().parents[2]
        self.assertEqual(values["manifest"], repo / "data/manifests/mtg_jamendo/mtg_jamendo_features.jsonl")
        self.assertEqual(values["output_dir"], repo / "data/features/mtg_jamendo/harmony")
        self.assertEqual(values["report"], repo / "outputs/mtg_jamendo/mert_chord_key/chords_keys_report.json")
        broken = self.root / "broken.yaml"
        broken.write_text(preset.read_text() + "unknown: true\n")
        with self.assertRaises(ConfigurationError):
            load_configuration(broken, "chord_key_extraction")

    def test_direct_entry_point_from_another_working_directory(self):
        repo = Path(__file__).resolve().parents[2]
        script = repo / "src/stemfusion/experiments/mtg_jamendo/extract_chords_keys.py"
        self.assertFalse((script.parent / "mert_chord_key/extract_chords_keys.py").exists())
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        result = subprocess.run([sys.executable, str(script), "--manifest", str(self.manifest),
            "--source", "mixture", "--split", "train", "--dry-run"], cwd=self.root,
            env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("planned=1", result.stderr)
        self.assertFalse(self.config.output_dir.exists())

    def test_unified_cli_routes_to_shared_dataset_entry(self):
        from stemfusion.cli import build_parser

        args = build_parser().parse_args(["extract", "mtg-jamendo", "chords-keys", "--manifest", str(self.manifest),
                                         "--source", "mixture", "--split", "train", "--dry-run"])
        self.assertEqual(args.command_handler.__module__, "stemfusion.experiments.mtg_jamendo.extract_chords_keys")
        self.assertEqual(args.source, ["mixture"])
        with patch("stemfusion.features.chords_keys._ChordKeyRuntime", side_effect=AssertionError("Dry run must not load BTC")):
            self.assertEqual(args.command_handler(args), 0)
        self.assertFalse(self.config.output_dir.exists())

    def test_retry_cli_switch_defaults_on_and_overrides_yaml(self):
        from stemfusion.cli import build_parser

        parser = build_parser()
        base = ["extract", "mtg-jamendo", "chords-keys", "--dry-run"]
        self.assertTrue(parser.parse_args(base).retry_failures)
        self.assertFalse(parser.parse_args(base + ["--no-retry-failures"]).retry_failures)
        self.assertTrue(parser.parse_args(base + ["--retry-failures"]).retry_failures)

    def test_output_directory_move_reuses_existing_files_without_rewriting(self):
        extract_chords_keys(self.config, runtime_factory=lambda _: self.runtime)
        before = {str(path.relative_to(self.config.output_dir)): path.read_bytes()
                  for path in self.config.output_dir.rglob("*") if path.is_file()}
        destination = self.root / "shared-harmony"
        self.config.output_dir.rename(destination)
        no_runtime = Mock(side_effect=AssertionError("A directory move must not require re-extraction"))
        report = extract_chords_keys(replace(self.config, output_dir=destination), runtime_factory=no_runtime)
        self.assertEqual((report["reused"], report["failed"], report["retry_attempted"]), (2, 0, 0))
        no_runtime.assert_not_called()
        self.assertEqual(before, {str(path.relative_to(destination)): path.read_bytes()
                                  for path in destination.rglob("*") if path.is_file()})


if __name__ == "__main__":
    unittest.main()

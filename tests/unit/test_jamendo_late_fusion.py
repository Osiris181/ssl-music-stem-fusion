"""Independent Jamendo branches, paired seeds, safe reuse, and saved ensembles."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tests.unit import test_jamendo_mid_fusion as fixtures
from stemfusion.cli import build_parser, main as cli_main
from stemfusion.configuration import ConfigurationError, default_config_path, load_configuration
from stemfusion.data.jamendo_late_fusion import prepare_late_fusion_data
from stemfusion.experiments import _mood_theme_tagging as training
from stemfusion.experiments.mtg_jamendo import reproduction, train_late
from stemfusion.configuration import JAMENDO_LATE_CONDITIONS as CONDITIONS
from stemfusion.experiments._mood_theme_tagging import (
    LateFusionConfig, branch_keys, format_late_summary as format_summary, run_late_fusion, train_branch,
)
from stemfusion.training import music2emo
from stemfusion.training.late_fusion import branch_seed
from stemfusion.training.music2emo_late_fusion import assemble_ensemble, member_identity, sum_predictions
from stemfusion.training.reporting import file_checksum


class JamendoLateFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.fixture = fixtures.JamendoMidFusionTests("test_all_sources_keep_baseline_identity_and_do_not_read_audio")
        self.fixture.setUp()
        self.root = self.fixture.root
        values = {item.name: getattr(self.fixture.config, item.name) for item in fields(reproduction.ReproductionConfig)}
        values["output_dir"] = self.root / "late"
        self.config = LateFusionConfig(**values, reference_output_dir=self.root / "reference")

    def tearDown(self):
        self.fixture.tearDown()

    def data(self, verify_checksums=True, sources=tuple(fixtures.SOURCES)):
        return prepare_late_fusion_data(self.config.manifest, self.config.mert_dir, self.config.harmony_dir,
                                        self.config.tag_list, self.config.author_metadata, sources,
                                        verify_checksums=verify_checksums)

    def test_full_plan_has_eleven_branches_and_twelve_reports_per_outer_seed(self):
        with patch("stemfusion.experiments._mood_theme_tagging.prepare_cached_fusion_data", side_effect=AssertionError("No features")):
            plan = run_late_fusion(replace(self.config, dry_run=True))
        self.assertEqual(plan["branches_per_outer_seed"], 11)
        self.assertEqual(plan["maximum_training_jobs"], 55)
        self.assertEqual(plan["report_conditions_per_seed"], 12)
        schedule = {row["branch_id"]: row for row in plan["branch_schedule"]["32"]}
        for index, source in enumerate(("mixture", "guitar", "drums", "piano", "bass", "vocals")):
            if index:
                self.assertEqual(schedule[source]["actual_training_seed"], schedule[f"mixture_{index + 1}"]["actual_training_seed"])
            self.assertEqual(schedule[f"mixture_{index + 1}"]["actual_training_seed"], branch_seed(32, index))
        self.assertEqual(schedule["guitar"]["actual_training_seed"], 2682760092)
        self.assertFalse(self.config.output_dir.exists())

    def test_selected_condition_only_requests_required_branches(self):
        for name, branches, sources in (("mixture_ensemble_4", 4, ["mixture"]),
                                        ("guitar_only", 1, ["mixture", "guitar"]),
                                        ("mixture_guitar_drums_piano", 4, ["mixture", "guitar", "drums", "piano"])):
            plan = run_late_fusion(replace(self.config, selected_run=name, dry_run=True))
            self.assertEqual(plan["branches_per_outer_seed"], branches)
            self.assertEqual(plan["sources"], sources)

    def test_configuration_is_inherited_strict_and_cli_overrides_replace_defaults(self):
        path = default_config_path("experiments/mtg_jamendo/late_fusion.yaml")
        defaults, provenance = load_configuration(path, "jamendo_late_fusion")
        self.assertEqual(defaults["runs"], CONDITIONS)
        self.assertEqual((defaults["epochs"], defaults["batch_size"], defaults["learning_rate"]), (200, 8, 1e-4))
        self.assertEqual((defaults["early_stopping_patience"], defaults["early_stopping_min_delta"]), (10, 1e-4))
        self.assertEqual(len(provenance["config_files"]), 3)
        self.assertTrue(defaults["verify_checksums"])
        parser = build_parser()
        args = parser.parse_args(["train", "mtg-jamendo", "late-fusion", "--seed", "64", "--seed", "128",
                                  "--run", "guitar_only", "--output-dir", str(self.root / "isolated"), "--dry-run"])
        config = training.config_from_args(args)
        self.assertEqual(config.seeds, (64, 128))
        self.assertEqual(config.selected_run, "guitar_only")
        self.assertEqual(config.summary_dir, self.root / "isolated")
        self.assertEqual(config.reference_output_dir, self.root / "isolated/reference")
        self.assertTrue(config.verify_checksums)
        args = parser.parse_args(["train", "mtg-jamendo", "late-fusion", "--no-verify-checksums"])
        self.assertFalse(training.config_from_args(args).verify_checksums)
        invalid = self.root / "invalid.yaml"
        invalid.write_text(f"training_profile: {path}\nruntime:\n  verify_checksums: false\n")
        defaults, _ = load_configuration(invalid, "jamendo_late_fusion")
        self.assertFalse(defaults["verify_checksums"])
        args = parser.parse_args(["train", "mtg-jamendo", "late-fusion", "--config", str(invalid), "--verify-checksums"])
        self.assertTrue(training.config_from_args(args).verify_checksums)
        invalid.write_text(f"training_profile: {path}\nruntime:\n  verify_checksums: 'false'\n")
        with self.assertRaises(ConfigurationError):
            load_configuration(invalid, "jamendo_late_fusion")
        invalid.write_text(f"training_profile: {path}\nfeatures:\n  fusion: average_probabilities\n")
        with self.assertRaises(ConfigurationError):
            load_configuration(invalid, "jamendo_late_fusion")
        invalid.write_text(f"training_profile: {path}\noptimization:\n  early_stopping:\n    monitor: pr_auc\n")
        with self.assertRaises(ConfigurationError):
            load_configuration(invalid, "jamendo_late_fusion")
        invalid.write_text(f"training_profile: {invalid}\n")
        with self.assertRaisesRegex(ConfigurationError, "Cyclic"):
            load_configuration(invalid, "jamendo_late_fusion")

    def test_direct_and_installed_commands_dry_run_without_pythonpath_or_writes(self):
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        process = subprocess.run([sys.executable, str(Path(train_late.__file__)), "--run", "guitar_only", "--dry-run"],
                                 cwd=self.root, env=env, text=True, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("no features/model loaded", process.stderr)
        self.assertEqual(cli_main(["train", "mtg-jamendo", "late-fusion", "--run", "guitar_only", "--dry-run"]), 0)
        self.assertFalse(self.config.output_dir.exists())

    def test_validation_only_does_not_train_or_write_outputs(self):
        with patch.object(music2emo, "fit_model", side_effect=AssertionError("No fitting")), patch.object(reproduction, "fit_seed", side_effect=AssertionError("No fitting")):
            result = run_late_fusion(replace(self.config, validate_only=True))
        self.assertTrue(result["validate_only"])
        self.assertFalse(self.config.output_dir.exists())
        self.assertFalse(self.config.reference_output_dir.exists())

    def test_checksum_modes_keep_exact_representations_and_declared_identities(self):
        for sources in (tuple(fixtures.SOURCES), ("mixture",), ("mixture", "guitar", "drums", "piano")):
            with self.subTest(sources=sources):
                retained = self.fixture.data(sources)
                verified, trusted = self.data(True, sources), self.data(False, sources)
                for candidate in (verified, trusted):
                    self.assertEqual(candidate.source_fingerprints, retained.source_fingerprints)
                    self.assertEqual(candidate.cohort_fingerprint, retained.cohort_fingerprint)
                    self.assertEqual(candidate.mixture.data_fingerprint, retained.mixture.data_fingerprint)
                    self.assertEqual(candidate.mixture.tag_names, retained.mixture.tag_names)
                    np.testing.assert_array_equal(candidate.mixture.prevalence, retained.mixture.prevalence)
                    for audit, expected in ((candidate.audit, retained.audit), (candidate.mixture.audit, retained.mixture.audit)):
                        self.assertEqual({key: value for key, value in audit.items() if key != "cache_verification"}, expected)
                    for split, records in retained.splits.items():
                        for first, second in zip(records, candidate.splits[split], strict=True):
                            self.assertEqual(first.track_id, second.track_id)
                            np.testing.assert_array_equal(first.labels, second.labels)
                            for source in sources:
                                a, b = first.views[source], second.views[source]
                                for name in ("segments", "roots", "qualities", "labels"):
                                    np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
                                self.assertEqual(a.mode, b.mode)
                self.assertTrue(verified.audit["cache_verification"]["feature_payload_checksums_verified"])
                self.assertFalse(trusted.audit["cache_verification"]["feature_payload_checksums_verified"])

    def test_only_trusted_mode_accepts_finite_changes_without_updating_markers(self):
        original = self.data()
        path = self.config.mert_dir / "mert_30s_guitar/01/1/segment_0.npy"
        array = np.load(path, allow_pickle=False)
        array[0, 5, 0] += 1
        np.save(path, array)
        with self.assertRaisesRegex(ValueError, "Changed or incomplete feature cache"):
            self.data()
        # This is the documented tradeoff: no claim of fresh byte-level identity.
        trusted = self.data(False)
        self.assertEqual(trusted.source_fingerprints, original.source_fingerprints)
        self.assertNotEqual(trusted.splits["train"][0].views["guitar"].segments[0, 0],
                            original.splits["train"][0].views["guitar"].segments[0, 0])

    def test_trusted_mode_still_rejects_invalid_arrays_and_incomplete_caches(self):
        path = self.config.mert_dir / "mert_30s_guitar/01/1/segment_0.npy"
        raw = path.read_bytes()
        for array in (np.full((1, 12, 768), np.nan, dtype=np.float32),
                      np.zeros((1, 12, 768), dtype=np.float64), np.zeros((12, 768), dtype=np.float32)):
            with self.subTest(shape=array.shape, dtype=array.dtype):
                np.save(path, array)
                with self.assertRaisesRegex(ValueError, "finite float32"):
                    self.data(False)
                path.write_bytes(raw)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing or incomplete feature cache"):
            self.data(False)
        path.write_bytes(raw)
        marker_path = path.parent / "run_manifest.json"
        marker_raw = marker_path.read_bytes()
        marker = json.loads(marker_raw)
        for change in ({"signature": ""}, {"files": {}}, {"files": {"../unsafe.npy": "fixture"}},
                       {"files": {"segment_0.npy": "fixture"}}):
            with self.subTest(change=change):
                marker_path.write_text(json.dumps({**marker, **change}))
                with self.assertRaises(ValueError):
                    self.data(False)
                marker_path.write_bytes(marker_raw)

    def test_trusted_mode_keeps_metadata_and_alignment_checks_mandatory(self):
        mert = self.config.mert_dir / "mert_30s_guitar/01/1/segments.json"
        harmony = self.config.harmony_dir / "guitar/01/1/metadata.json"
        mixture = self.config.mert_dir / "mert_30s/01/1/segments.json"
        changes = [(mert, ("track_id",), "wrong"), (mert, ("split",), "test"),
                   (mert, ("source",), "bass"), (mert, ("audio", "path"), "wrong"),
                   (mert, ("reference_mixture", "size"), 999),
                   (mert, ("settings", "minimum_input_samples"), 999),
                   (mert, ("segments", 0, "end_sample"), 111),
                   (mert, ("segments", 0, "file"), "segment_1.npy"),
                   (harmony, ("settings", "normalization"), "wrong"),
                   (harmony, ("settings", "checkpoint_sha256"), "wrong"),
                   (mixture, ("settings", "revision"), "wrong")]
        for path, keys, value in changes:
            with self.subTest(path=path, keys=keys):
                raw = path.read_bytes()
                metadata = json.loads(raw)
                entry = metadata
                for key in keys[:-1]:
                    entry = entry[key]
                entry[keys[-1]] = value
                path.write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):
                    self.data(False)
                path.write_bytes(raw)
        np.save(self.config.author_metadata / "probas_train.npy", np.full(56, .4, dtype=np.float64))
        with self.assertRaisesRegex(ValueError, "class weights do not match"):
            self.data(False)

    def test_trusted_validation_and_training_record_policy_without_changing_training_or_reuse(self):
        config = replace(self.config, seeds=(32,), selected_run="guitar_only", verify_checksums=False)
        with patch.object(music2emo, "fit_model", side_effect=AssertionError("No fitting")), self.assertLogs(
                "stemfusion.experiments._mood_theme_tagging", level="WARNING") as messages:
            report = run_late_fusion(replace(config, validate_only=True))
        self.assertIn("may go undetected", messages.output[0])
        self.assertFalse(report["verify_checksums"])
        self.assertFalse(report["audit"]["cache_verification"]["feature_payload_checksums_verified"])
        self.assertFalse(config.output_dir.exists())
        with contextlib.redirect_stdout(io.StringIO()):
            trusted = run_late_fusion(config)
        with patch.object(music2emo, "set_seed", side_effect=AssertionError("Reuse branch")), contextlib.redirect_stdout(io.StringIO()):
            checked = run_late_fusion(replace(config, verify_checksums=True))
        self.assertEqual(trusted["conditions"], checked["conditions"])
        self.assertFalse(trusted["data_audit"]["cache_verification"]["feature_payload_checksums_verified"])
        self.assertTrue(checked["data_audit"]["cache_verification"]["feature_payload_checksums_verified"])
        with contextlib.redirect_stdout(io.StringIO()):
            fresh = run_late_fusion(replace(config, verify_checksums=True, output_dir=self.root / "late_checked"))
        first = Path(trusted["runs"]["guitar_only"][32]["checkpoint"]).parent / "predictions_test.npz"
        second = Path(fresh["runs"]["guitar_only"][32]["checkpoint"]).parent / "predictions_test.npz"
        with np.load(first) as a, np.load(second) as b:
            np.testing.assert_array_equal(a["logits"], b["logits"])
        with self.assertRaisesRegex(ValueError, "boolean"):
            run_late_fusion(replace(config, verify_checksums="false"))

    def test_five_seed_matrix_reuses_reference_and_reports_stems_without_extra_training(self):
        data = self.fixture.data()
        reference_values = {item.name: getattr(self.config, item.name) for item in fields(reproduction.ReproductionConfig)}
        reference_values["output_dir"] = self.config.reference_output_dir
        reference_config = reproduction.ReproductionConfig(**reference_values)
        for seed in self.config.seeds:
            reproduction.fit_seed(reference_config, data.mixture, seed)
        original = {path: file_checksum(path) for path in self.config.reference_output_dir.rglob("*") if path.is_file()}
        with patch.object(reproduction, "Music2EmoClassifier", side_effect=AssertionError("No reference retraining")), patch.object(music2emo, "fit_model", wraps=music2emo.fit_model) as fitting, contextlib.redirect_stdout(io.StringIO()):
            summary = run_late_fusion(self.config)
        self.assertEqual(fitting.call_count, 50)
        self.assertEqual(set(summary["conditions"]), set(CONDITIONS))
        self.assertEqual(len(summary["standalone_conditions"]), 5)
        self.assertEqual(len(summary["paired_differences"]), 3)
        self.assertEqual(len(summary["branches"]), 5)
        self.assertFalse(summary["significance_tests_performed"])
        for seed in self.config.seeds:
            branches = summary["branches"][seed]
            self.assertEqual(len(branches), 11)
            self.assertEqual(len({row["checkpoint"] for row in branches.values()}), 11)
            for source in ("guitar", "drums", "piano", "bass", "vocals"):
                standalone = summary["runs"][source + "_only"][seed]
                self.assertEqual(standalone["checkpoint"], branches[source]["checkpoint"])
                self.assertEqual(standalone["test"], branches[source]["test"])
                self.assertEqual(standalone["threshold"], branches[source]["threshold"])
                self.assertEqual(standalone["additional_training_jobs"], 0)
                position = branches[source]["index"]
                self.assertEqual(branches[source]["actual_training_seed"], branches[f"mixture_{position + 1}"]["actual_training_seed"])
            for name in ("mixture_guitar_drums_piano", "mixture_ensemble_4"):
                result = summary["runs"][name][seed]
                members = [branches[row["branch_id"]] for row in result["members"]]
                expected = sum_predictions(members, "test")
                with np.load(Path(result["output_directory"]) / "predictions_test.npz") as saved:
                    np.testing.assert_array_equal(saved["logits"], expected["logits"])
                    np.testing.assert_array_equal(saved["probabilities"], expected["probabilities"])
                validation = sum_predictions(members, "validation")
                self.assertEqual(result["threshold"], music2emo.select_threshold(validation))
                self.assertEqual(result["trainable_parameters"], 4 * 940216)
        self.assertEqual(summary["runs"]["mixture_only"][32]["actual_training_seed"], 32)
        self.assertTrue(all(file_checksum(path) == checksum for path, checksum in original.items()))
        readable = (self.config.summary_dir / "summary_late_fusion.txt").read_text()
        self.assertIn("Supporting stem-only results", readable)
        self.assertIn("vocals_only", readable)
        self.assertEqual(readable, format_summary(summary))
        metric = summary["conditions"]["guitar_only"]["pr_auc_macro"]
        self.assertAlmostEqual(metric["sample_sd"], float(np.std(metric["values"], ddof=1)))
        with patch.object(music2emo, "set_seed", side_effect=AssertionError("No branch training")), patch.object(reproduction, "Music2EmoClassifier", side_effect=AssertionError("No reference training")), contextlib.redirect_stdout(io.StringIO()):
            again = run_late_fusion(self.config)
        self.assertEqual(summary["conditions"], again["conditions"])
        full_summary = file_checksum(self.config.summary_dir / "summary_late_fusion.json")
        with patch.object(music2emo, "set_seed", side_effect=AssertionError("Reuse stem branch")), contextlib.redirect_stdout(io.StringIO()):
            selected = run_late_fusion(replace(self.config, seeds=(32,), selected_run="guitar_only"))
        self.assertEqual(selected["seeds"], [32])
        self.assertTrue((self.config.summary_dir / "summary_late_fusion_run_guitar_only_seeds_32.txt").is_file())
        self.assertEqual(file_checksum(self.config.summary_dir / "summary_late_fusion.json"), full_summary)
        with self.assertRaisesRegex(ValueError, "incompatible|mismatch"):
            run_late_fusion(replace(self.config, epochs=2, seeds=(32,), selected_run="mixture_only"))

    def test_independent_mixture_branch_matches_reproduction_at_its_actual_seed(self):
        config = replace(self.config, seeds=(32,), epochs=2)
        data = self.fixture.data(("mixture",))
        actual = branch_seed(32, 1)
        branch = train_branch(config, data, outer_seed=32, source="mixture", index=1, implementation="fixture")
        values = {item.name: getattr(config, item.name) for item in fields(reproduction.ReproductionConfig)}
        values["output_dir"] = self.root / "parity_reference"
        original = reproduction.fit_seed(reproduction.ReproductionConfig(**values), data.mixture, actual)
        with np.load(Path(branch["output_directory"]) / "predictions_test.npz") as first, np.load(Path(original["checkpoint"]).parent / "predictions_test.npz") as second:
            np.testing.assert_array_equal(first["logits"], second["logits"])
        self.assertEqual(branch["best_epoch"], original["best_epoch"])

    def test_ensemble_rejects_repeated_seeds_checkpoints_and_cohort_changes(self):
        branches = [{"actual_training_seed": index, "checkpoint": str(self.root / f"{index}.pt"),
                     "outer_seed": 32, "cohort_fingerprint": "same"} for index in range(4)]
        kwargs = dict(condition="fixture", outer_seed=32, output_directory=self.root / "ensemble",
                      prevalence=np.full(56, .5), implementation="fixture")
        changed = [dict(row) for row in branches]
        changed[1]["actual_training_seed"] = 0
        with self.assertRaisesRegex(ValueError, "distinct actual"):
            assemble_ensemble(changed, **kwargs)
        changed = [dict(row) for row in branches]
        changed[1]["checkpoint"] = changed[0]["checkpoint"]
        with self.assertRaisesRegex(ValueError, "single trained checkpoint"):
            assemble_ensemble(changed, **kwargs)
        changed = [dict(row) for row in branches]
        changed[1]["cohort_fingerprint"] = "different"
        with self.assertRaisesRegex(ValueError, "cohort"):
            assemble_ensemble(changed, **kwargs)

    def test_prediction_sum_rejects_id_or_target_misalignment(self):
        branches = []
        for index in range(2):
            directory = self.root / f"branch_{index}"
            directory.mkdir()
            np.savez(directory / "predictions_test.npz", track_ids=np.asarray(["a", "b"]),
                     targets=np.tile(np.asarray([[0], [1]], dtype=np.uint8), (1, 56)),
                     logits=np.full((2, 56), index + 1, dtype=np.float32))
            branches.append({"output_directory": str(directory)})
        result = sum_predictions(branches, "test")
        np.testing.assert_array_equal(result["logits"], np.full((2, 56), 3, dtype=np.float32))
        np.testing.assert_allclose(result["probabilities"], torch.tensor(3.).sigmoid().numpy())
        path = Path(branches[1]["output_directory"]) / "predictions_test.npz"
        np.savez(path, track_ids=np.asarray(["b", "a"]), targets=result["targets"], logits=result["logits"])
        with self.assertRaisesRegex(ValueError, "IDs, ordering"):
            sum_predictions(branches, "test")

    def test_changed_prediction_artifacts_are_not_reused(self):
        data = self.fixture.data(("mixture",))
        branch = train_branch(self.config, data, outer_seed=32, source="mixture", index=1, implementation="fixture")
        member_identity(branch)
        (Path(branch["output_directory"]) / "predictions_test.npz").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "artifact is missing or changed"):
            member_identity(branch)


if __name__ == "__main__":
    unittest.main()

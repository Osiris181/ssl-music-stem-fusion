"""EmoMusic native-score late fusion with the exact PMEmo regression matrix."""

from __future__ import annotations

import ast
import contextlib
import inspect
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

from tests.unit import test_emomusic_mid_fusion as fixtures
from tests.unit.test_pmemo_mid_fusion import TinyFusion
from stemfusion.cli import build_parser
from stemfusion.configuration import ConfigurationError, default_config_path, load_configuration
from stemfusion.experiments import _emomusic_late_fusion as controller
from stemfusion.experiments import _emomusic_mid_fusion as mid
from stemfusion.experiments import _emotion_late_fusion as pmemo_late, _emotion_recognition as pmemo_mid
from stemfusion.experiments.emomusic import train_late
from stemfusion.training import emotion_late_fusion as ensemble, music2emo
from stemfusion.training.emotion import BinaryEmotionTask
from stemfusion.training.late_fusion import branch_seed
from stemfusion.training.reporting import file_checksum


class EmoMusicLateFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.fixture = fixtures.EmoMusicMidFusionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        values = {item.name: getattr(self.fixture.config, item.name) for item in fields(mid.EmotionMidFusionConfig)}
        values.update(output_dir=self.root / "late")
        values.pop("conditions")
        self.config = controller.EmotionLateFusionConfig(**values, reference_output_dir=self.fixture.config.output_dir)

    def tiny_models(self):
        stack = contextlib.ExitStack()
        for module in (controller, mid):
            stack.enter_context(patch.object(module, "Music2EmoEmotionMidFusion", TinyFusion))
            stack.enter_context(patch.object(module, "parameter_count", side_effect=lambda count: count * 1536 + 1))
        return stack

    def train(self, config=None, features=None):
        with self.tiny_models(), contextlib.redirect_stdout(io.StringIO()):
            return controller.run_late_fusion(config or self.config, features=features or self.fixture.features())

    def test_core_scheduling_and_branch_fitting_are_ast_identical_to_pmemo(self):
        source = inspect.getsource(controller._plan).replace("MID_CONDITIONS", "mid.MID_CONDITIONS")
        self.assertEqual(ast.dump(ast.parse(source)), ast.dump(ast.parse(inspect.getsource(pmemo_late._plan))))
        source = inspect.getsource(controller.train_branch)
        source = source.replace('    label_policy = label_policy_for(config.dataset)\n', '')
        source = source.replace('mid.implementation_identity(config.dataset)', 'mid.implementation_identity()')
        source = source.replace('"label_policy": label_policy', '"label_policy": LABEL_POLICY')
        source = source.replace('"%s %s | outer seed %d | branch %s | training seed %d", config.dataset.upper(), target,',
                                '"PMEmo %s | outer seed %d | branch %s | training seed %d", target,')
        self.assertEqual(ast.dump(ast.parse(source)), ast.dump(ast.parse(inspect.getsource(pmemo_late.train_branch))))
        for name in ("assemble_ensemble", "branch_seed", "branch_id", "branch_keys", "BinaryEmotionLateTask",
                     "BinaryEmotionTask", "JamendoMidFusionDataset", "Music2EmoEmotionMidFusion", "music2emo"):
            self.assertIs(getattr(controller, name), getattr(pmemo_late, name), name)

    def test_native_late_algorithms_are_ast_identical_to_deam(self):
        from stemfusion.experiments import _emotion_native_late_fusion as deam
        for name in ("_plan", "train_branch", "format_late_summary", "run_late_fusion", "config_from_args"):
            self.assertEqual(ast.dump(ast.parse(inspect.getsource(getattr(controller, name)))),
                             ast.dump(ast.parse(inspect.getsource(getattr(deam, name)))), name)
        for name in ("assemble_ensemble", "branch_seed", "branch_id", "branch_keys", "BinaryEmotionLateTask",
                     "BinaryEmotionTask", "JamendoMidFusionDataset", "Music2EmoEmotionMidFusion", "music2emo"):
            self.assertIs(getattr(controller, name), getattr(deam, name), name)

    def test_equivalent_native_and_pmemo_inputs_produce_identical_branch_and_ensemble_predictions(self):
        conditions = {name: controller.LATE_CONDITIONS[name] for name in
                      ("mixture_ensemble_4", "mixture_guitar_drums_piano")}
        native = replace(self.config, targets=("valence",), seeds=(32,), conditions=conditions)
        original_fixture = self.fixture.fixture
        values = {item.name: getattr(original_fixture.config, item.name) for item in fields(pmemo_mid.EmotionMidFusionConfig)}
        values.update(targets=("valence",), seeds=(32,), conditions=conditions,
                      output_dir=self.root / "pmemo_late")
        previous = pmemo_late.EmotionLateFusionConfig(**values, reference_output_dir=original_fixture.config.output_dir)
        with self.tiny_models(), contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()):
            for module in (pmemo_late, pmemo_mid):
                stack.enter_context(patch.object(module, "Music2EmoEmotionMidFusion", TinyFusion))
                stack.enter_context(patch.object(module, "parameter_count", side_effect=lambda count: count * 1536 + 1))
            current = controller.run_late_fusion(native, features=self.fixture.features())["targets"]["valence"]
            prior = pmemo_late.run_late_fusion(previous, features=original_fixture.features())["targets"]["valence"]
        self.assertEqual(current["conditions"], prior["conditions"])
        self.assertEqual(current["paired_differences"], prior["paired_differences"])
        pairs = [(branch, prior["branches"][32][name]) for name, branch in current["branches"][32].items()]
        pairs += [(current["runs"][name][32], prior["runs"][name][32]) for name in conditions]
        for first, second in pairs:
            self.assertEqual(first["test"], second["test"])
            self.assertEqual(first["threshold"], second["threshold"])
            for split in ("validation", "test"):
                with np.load(Path(first["output_directory"]) / f"predictions_{split}.npz") as a, \
                        np.load(Path(second["output_directory"]) / f"predictions_{split}.npz") as b:
                    for key in ("track_ids", "targets", "logits", "probabilities", "predictions"):
                        np.testing.assert_array_equal(a[key], b[key])

    def test_full_plan_matches_jamendo_matrix_and_position_paired_seeds_without_io(self):
        with patch.object(controller, "prepare_emotion_features", side_effect=AssertionError("No features")), \
                contextlib.redirect_stdout(io.StringIO()):
            plan = controller.run_late_fusion(replace(self.config, dry_run=True))
        self.assertEqual(plan["branches_per_target_outer_seed"], 11)
        self.assertEqual(plan["report_conditions_per_target_seed"], 12)
        self.assertEqual(plan["maximum_training_jobs"], 110)
        self.assertEqual(plan["maximum_new_branch_training_jobs"], 100)
        self.assertEqual(plan["maximum_reference_training_jobs"], 10)
        schedule = {row["branch_id"]: row for row in plan["branch_schedule"]["32"]}
        for index, source in enumerate(fixtures.SOURCES):
            self.assertEqual(schedule[f"mixture_{index + 1}"]["actual_training_seed"], branch_seed(32, index))
            if index:
                self.assertEqual(schedule[source]["actual_training_seed"], schedule[f"mixture_{index + 1}"]["actual_training_seed"])
        self.assertFalse(self.config.output_dir.exists())

    def test_condition_selection_invalid_inputs_and_reference_path_safety(self):
        for name, count, sources in (("guitar_only", 1, ["mixture", "guitar"]),
                                     ("mixture_ensemble_4", 4, ["mixture"]),
                                     ("mixture_guitar_drums_piano", 4, list(fixtures.SOURCES[:4]))):
            plan, _ = controller._plan(replace(self.config, selected_run=name))
            self.assertEqual(plan["branches_per_target_outer_seed"], count)
            self.assertEqual(plan["sources"], sources)
        for changes in ({"targets": ("invalid",)}, {"seeds": (32, 32)}, {"verify_checksums": "false"},
                        {"selected_run": "invalid"}, {"reference_output_dir": None},
                        {"reference_output_dir": self.config.output_dir},
                        {"reference_output_dir": self.config.output_dir / "reference"},
                        {"reference_output_dir": self.root}, {"learning_rate": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                controller._plan(replace(self.config, **changes))

    def test_yaml_settings_match_mid_and_jamendo_and_reject_protocol_changes(self):
        preset = default_config_path("experiments/emomusic/late_fusion.yaml")
        defaults, provenance = load_configuration(preset, "emomusic_late_fusion")
        prior, _ = load_configuration(default_config_path("experiments/emomusic/mid_fusion.yaml"), "emomusic_mid_fusion")
        jamendo, _ = load_configuration(default_config_path("experiments/mtg_jamendo/late_fusion.yaml"), "jamendo_late_fusion")
        for key in ("epochs", "batch_size", "learning_rate", "early_stopping_patience", "early_stopping_min_delta",
                    "seeds", "device", "num_workers", "verify_checksums"):
            self.assertEqual(defaults[key], prior[key], key)
            self.assertEqual(defaults[key], jamendo[key], key)
        self.assertEqual(defaults["runs"], jamendo["runs"])
        self.assertEqual(len(provenance["config_files"]), 3)
        args = build_parser().parse_args(["train", "emomusic", "late-fusion", "--target", "arousal", "--seed", "64",
            "--run", "guitar_only", "--no-verify-checksums", "--output-dir", str(self.root / "isolated"), "--dry-run"])
        config = controller.config_from_args(args)
        self.assertEqual((config.targets, config.seeds, config.selected_run), (("arousal",), (64,), "guitar_only"))
        self.assertFalse(config.verify_checksums)
        self.assertEqual(config.summary_dir, self.root / "isolated")
        self.assertEqual(config.reference_output_dir, self.root / "isolated_reference")
        invalid = self.root / "invalid.yaml"
        for override in ("features:\n  fusion: average_probabilities\n", "task:\n  label_threshold: 0.6\n",
                         "runtime:\n  verify_checksums: 'false'\n", "model:\n  output_labels: 56\n",
                         "optimization:\n  early_stopping:\n    monitor: accuracy\n"):
            invalid.write_text(f"training_profile: {preset}\n" + override)
            with self.assertRaises(ConfigurationError):
                load_configuration(invalid, "emomusic_late_fusion")

    def test_direct_command_outside_repository_does_not_load_features_or_write_outputs(self):
        environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        result = subprocess.run([sys.executable, str(Path(train_late.__file__)), "--dry-run", "--target", "valence",
                                 "--output-dir", str(self.root / "isolated")],
                                cwd=self.root, env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["targets"], ["valence"])
        self.assertFalse((self.root / "isolated").exists())
        with patch.object(controller, "run_late_fusion") as run:
            self.assertEqual(train_late.main(["--dry-run"]), 0)
        self.assertEqual(run.call_args.args[0].conditions, controller.LATE_CONDITIONS)

    def test_validation_only_keeps_all_outputs_absent_and_never_constructs_model(self):
        for verify in (True, False):
            with patch.object(music2emo, "fit_model", side_effect=AssertionError("No fitting")), \
                    patch.object(controller, "Music2EmoEmotionMidFusion", side_effect=AssertionError("No model")), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = controller.run_late_fusion(replace(self.config, validate_only=True, verify_checksums=verify))
            self.assertTrue(result["validate_only"])
            self.assertEqual(set(result["targets"]), {"valence", "arousal"})
        self.assertFalse(self.config.output_dir.exists())
        self.assertFalse(self.config.reference_output_dir.exists())
        self.assertFalse(self.config.summary_dir.exists())

    def test_full_five_seed_two_target_matrix_reuses_mid_references_and_stems_without_extra_fits(self):
        features = self.fixture.features()
        with self.tiny_models(), contextlib.redirect_stdout(io.StringIO()):
            references = mid.run_mid_fusion(replace(self.fixture.config, selected_run="mixture_only"), features=features)
            prior = {path: file_checksum(path) for path in self.fixture.config.output_dir.rglob("*") if path.is_file()}
            prior.update({path: file_checksum(path) for path in self.config.summary_dir.glob("summary_mid*")})
            with patch.object(controller, "Music2EmoEmotionMidFusion", wraps=TinyFusion) as fitting:
                result = controller.run_late_fusion(self.config, features=features)
            self.assertEqual(fitting.call_count, 100)
            complete = {path: file_checksum(path) for path in self.config.output_dir.rglob("*") if path.is_file()}
            with patch.object(music2emo, "set_seed", side_effect=AssertionError("No retraining")):
                reused = controller.run_late_fusion(self.config, features=features)
        self.assertEqual(len(list(self.config.output_dir.rglob("best.pt"))), 100)
        self.assertTrue(all(file_checksum(path) == sha for path, sha in {**prior, **complete}.items()))
        for target in fixtures.controller.TARGETS:
            summary = result["targets"][target]
            self.assertEqual(len(summary["conditions"]), 12)
            self.assertEqual(len(summary["paired_differences"]), 3)
            self.assertEqual(summary["seeds"], [32, 64, 128, 256, 512])
            self.assertEqual(summary["label_policy"]["threshold"], 5.0)
            self.assertEqual(summary["dataset"], "emomusic")
            self.assertEqual(summary["conditions"], reused["targets"][target]["conditions"])
            self.assertEqual(summary["conditions"]["mixture_only"], references["targets"][target]["conditions"]["mixture_only"])
            for source in fixtures.SOURCES[1:]:
                for seed in mid.SEEDS:
                    standalone = summary["runs"][f"{source}_only"][seed]
                    branch = summary["branches"][seed][source]
                    self.assertEqual(standalone["checkpoint"], branch["checkpoint"])
                    self.assertEqual(standalone["test"], branch["test"])
                    self.assertEqual(standalone["additional_training_jobs"], 0)
            text = (self.config.summary_dir / f"summary_late_fusion_{target}.txt").read_text()
            self.assertIn("Supporting stem-only results", text)
            self.assertIn("vocals_only", text)
            self.assertIn("mean +/- sample SD (ddof=1)", text)
            self.assertIn("original 1-9 score >= 5", text)
            self.assertIn("No score normalization", text)
            self.assertTrue((self.config.summary_dir / f"summary_late_fusion_{target}.json").is_file())

    def test_real_binary_architecture_can_fit_a_stem_branch(self):
        config = replace(self.config, targets=("valence",), seeds=(32,), selected_run="guitar_only")
        with contextlib.redirect_stdout(io.StringIO()):
            result = controller.run_late_fusion(config, features=self.fixture.features())
        branch = result["targets"]["valence"]["runs"]["guitar_only"][32]
        self.assertEqual(branch["trainable_parameters"], controller.parameter_count(1))
        self.assertEqual(branch["best_epoch"], 1)
        self.assertEqual(branch["actual_training_seed"], branch_seed(32, 1))

    def test_logit_sum_not_probability_mean_and_validation_only_threshold(self):
        config = replace(self.config, targets=("valence",), seeds=(32,), selected_run="mixture_ensemble_4")
        result = self.train(config)
        summary = result["targets"]["valence"]
        branches = list(summary["branches"][32].values())
        arrays = ensemble.sum_predictions(branches, "test")
        raw, probabilities = [], []
        for branch in branches:
            with np.load(Path(branch["output_directory"]) / "predictions_test.npz") as saved:
                raw.append(saved["logits"])
                probabilities.append(saved["probabilities"])
        np.testing.assert_array_equal(arrays["logits"], np.sum(np.stack(raw), axis=0, dtype=np.float32))
        np.testing.assert_array_equal(arrays["probabilities"], torch.from_numpy(arrays["logits"]).sigmoid().numpy())
        self.assertFalse(np.allclose(arrays["probabilities"], np.mean(probabilities, axis=0)))
        fit = summary["runs"]["mixture_ensemble_4"][32]
        expected = BinaryEmotionTask("valence").select_threshold(ensemble.sum_predictions(branches, "validation"))
        self.assertEqual(fit["threshold"], expected)
        for split in ("validation", "test"):
            with np.load(Path(fit["output_directory"]) / f"predictions_{split}.npz") as saved:
                self.assertEqual(saved["threshold"].item(), expected)

    def test_repeated_members_mixed_targets_seed_or_cohort_and_tampered_checkpoint_are_rejected(self):
        result = self.train(replace(self.config, targets=("valence",), seeds=(32,), selected_run="mixture_ensemble_4"))
        branches = list(result["targets"]["valence"]["branches"][32].values())
        def assemble(members):
            return ensemble.assemble_ensemble(members, condition="mixture_ensemble_4", outer_seed=32,
                output_directory=self.root / "checked", prevalence=np.array([0.5]), target="valence", implementation="fixture")
        invalid = [branches[:3] + [branches[0]],
                   [*branches[:3], {**branches[3], "actual_training_seed": branches[0]["actual_training_seed"]}],
                   [*branches[:3], {**branches[3], "checkpoint": branches[0]["checkpoint"]}],
                   [*branches[:3], {**branches[3], "outer_seed": 64}],
                   [*branches[:3], {**branches[3], "cohort_fingerprint": "other"}],
                   [*branches[:3], {**branches[3], "protocol": {"target": "arousal"}}]]
        for members in invalid:
            with self.assertRaises(ValueError):
                assemble(members)
        checkpoint = Path(branches[-1]["checkpoint"])
        checkpoint.write_bytes(b"changed checkpoint")
        with self.assertRaisesRegex(ValueError, "checkpoint is missing or changed"):
            assemble(branches)

    def test_misaligned_ids_targets_shapes_and_nonfinite_logits_are_rejected(self):
        result = self.train(replace(self.config, targets=("valence",), seeds=(32,), selected_run="mixture_ensemble_4"))
        branches = list(result["targets"]["valence"]["branches"][32].values())
        path = Path(branches[-1]["output_directory"]) / "predictions_test.npz"
        with np.load(path) as saved:
            original = {key: saved[key].copy() for key in saved.files}
        for changes in ({"track_ids": original["track_ids"][::-1]}, {"targets": 1 - original["targets"]},
                        {"logits": np.zeros((2, 56), dtype=np.float32)},
                        {"logits": np.full((2, 1), np.nan, dtype=np.float32)}):
            music2emo._save_predictions(path, {**original, **changes})
            with self.assertRaises(ValueError):
                ensemble.sum_predictions(branches, "test")
        self.assertFalse((self.root / "checked").exists())

    def test_interrupted_branch_resume_is_exact_and_changed_settings_fail(self):
        features = self.fixture.features()
        config = replace(self.config, targets=("valence",), seeds=(32,), epochs=2, selected_run="guitar_only")
        original_fit = music2emo.fit_model
        def interrupted_fit(*args, **kwargs):
            def stop_after_one(epoch):
                raise RuntimeError("interrupted")
            return original_fit(*args, **kwargs, epoch_callback=stop_after_one)
        with self.tiny_models(), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(music2emo, "fit_model", side_effect=interrupted_fit), self.assertRaisesRegex(RuntimeError, "interrupted"):
                controller.run_late_fusion(config, features=features)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                controller.run_late_fusion(replace(config, learning_rate=2e-4), features=features)
            resumed = controller.run_late_fusion(config, features=features)
            fresh = controller.run_late_fusion(replace(config, output_dir=self.root / "fresh"), features=features)
        a = resumed["targets"]["valence"]["runs"]["guitar_only"][32]
        b = fresh["targets"]["valence"]["runs"]["guitar_only"][32]
        self.assertEqual(a["test"], b["test"])
        self.assertEqual(a["best_epoch"], b["best_epoch"])
        with np.load(Path(a["output_directory"]) / "predictions_test.npz") as saved_a, \
                np.load(Path(b["output_directory"]) / "predictions_test.npz") as saved_b:
            np.testing.assert_array_equal(saved_a["logits"], saved_b["logits"])

    def test_overwrite_does_not_replace_reference_and_changed_reference_settings_are_rejected(self):
        features = self.fixture.features()
        config = replace(self.config, targets=("valence",), seeds=(32,), selected_run="mixture_only")
        first = self.train(config, features)
        branch = first["targets"]["valence"]["branches"][32]["mixture_1"]
        directory = Path(branch["output_directory"])
        before = {path: file_checksum(path) for path in directory.rglob("*") if path.is_file()}
        with self.tiny_models(), patch.object(music2emo, "set_seed", side_effect=AssertionError("No retraining")), \
                contextlib.redirect_stdout(io.StringIO()):
            controller.run_late_fusion(replace(config, overwrite=True), features=features)
        self.assertTrue(all(file_checksum(path) == sha for path, sha in before.items()))
        with self.assertRaisesRegex(ValueError, "incompatible|mismatch"):
            self.train(replace(config, learning_rate=2e-4), features)


if __name__ == "__main__":
    unittest.main()

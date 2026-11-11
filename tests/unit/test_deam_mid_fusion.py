"""Native DEAM scores through PMEmo-equivalent loading, fitting and reporting."""

from __future__ import annotations

import ast
import contextlib
import copy
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

from tests.unit import test_pmemo_mid_fusion as fixtures
from stemfusion.cli import build_parser
from stemfusion.configuration import ConfigurationError, default_config_path, load_configuration
from stemfusion.data import emotion_fusion, emotion_mid_fusion
from stemfusion.data.jamendo_mid_fusion import SOURCES, JamendoMidFusionDataset
from stemfusion.experiments import _emotion_mid_fusion as controller, _emotion_recognition as pmemo
from stemfusion.experiments.deam import train_mid
from stemfusion.training import music2emo
from stemfusion.training.reporting import file_checksum, write_json


class DEAMMidFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.fixture = fixtures.PMEmoMidFusionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        values = {item.name: getattr(self.fixture.config, item.name) for item in fields(pmemo.EmotionMidFusionConfig)}
        values.update(manifest=self.root / "deam.jsonl", output_dir=self.root / "deam_runs", summary_dir=self.root / "deam_summaries")
        self.config = controller.EmotionMidFusionConfig(**values)
        self.rows = copy.deepcopy(self.fixture.rows)
        for row in self.rows:
            row["dataset"] = "deam"
            # The synthetic DEAM scores retain the same midpoint labels only.
            # Production code never converts or replaces native scores.
            row["annotations"] = {target: 1 + 8 * value for target, value in row["annotations"].items()}
        self.write_manifest()

    def write_manifest(self):
        self.config.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def features(self, **kwargs):
        return emotion_fusion.prepare_emotion_features(self.config.manifest, self.config.mert_dir,
            self.config.harmony_dir, self.config.chord_root, self.config.chord_attr,
            adapter=controller.dataset_adapter("deam"), **kwargs)

    def tiny_models(self):
        stack = contextlib.ExitStack()
        for module in (controller, pmemo):
            stack.enter_context(patch.object(module, "Music2EmoEmotionMidFusion", fixtures.TinyFusion))
            stack.enter_context(patch.object(module, "parameter_count", side_effect=lambda count: count * 1536 + 1))
        return stack

    def test_native_midpoint_includes_equality_and_does_not_change_scores_or_manifest(self):
        original = self.config.manifest.read_bytes()
        features = self.features()
        self.assertEqual(features.annotations["2"]["valence"], 5.0)
        for target, expected in (("valence", [0, 1]), ("arousal", [1, 0])):
            data = features.target(target)
            self.assertEqual([int(record.labels[0]) for record in data.splits["train"]], expected)
            self.assertEqual(data.audit["label_policy"]["threshold"], 5.0)
            self.assertEqual(data.audit["label_policy"]["annotation_scale"], "supplied_1_9")
        self.assertEqual(self.config.manifest.read_bytes(), original)
        self.assertEqual(features.annotations["2"]["valence"], 5.0)
        with self.assertRaises(ValueError):
            features.target("unknown")
        with self.assertRaises(ValueError):
            emotion_fusion.label_policy_for("emomusic")  # Unreviewed dataset rule stays disabled.

    def test_native_labels_match_normalized_midpoint_but_original_score_is_in_cohort_identity(self):
        features = self.features()
        data = features.target("valence")
        for split, records in data.splits.items():
            for record in records:
                score = features.annotations[record.track_id]["valence"]
                self.assertEqual(int(record.labels[0]), int((score - 1) / 8 >= 0.5))
        features.annotations["1"]["valence"] = 3.5
        changed = features.target("valence")
        self.assertNotEqual(data.cohort_fingerprint, changed.cohort_fingerprint)
        np.testing.assert_array_equal(data.splits["train"][0].labels, changed.splits["train"][0].labels)

    def test_representations_encoding_sampling_and_labels_match_frozen_pmemo(self):
        deam, previous = self.features(), self.fixture.features()
        for target in ("valence", "arousal"):
            a, b = deam.target(target), previous.target(target)
            self.assertNotEqual(a.cohort_fingerprint, b.cohort_fingerprint)
            np.testing.assert_array_equal(a.mixture.prevalence, b.mixture.prevalence)
            for split in a.splits:
                for first, second in zip(a.splits[split], b.splits[split], strict=True):
                    np.testing.assert_array_equal(first.labels, second.labels)
                    for source in SOURCES:
                        for name in ("segments", "roots", "qualities", "labels"):
                            np.testing.assert_array_equal(getattr(first.views[source], name), getattr(second.views[source], name))
                        self.assertEqual(first.views[source].mode, second.views[source].mode)
            for training in (False, True):
                samples = []
                for data in (a, b):
                    np.random.seed(32)
                    samples.append(JamendoMidFusionDataset(data.splits["train"], sources=SOURCES, training=training)[0])
                for name in samples[0]:
                    if isinstance(samples[0][name], str):
                        self.assertEqual(samples[0][name], samples[1][name])
                    else:
                        torch.testing.assert_close(samples[0][name], samples[1][name], rtol=0, atol=0)

    def test_cache_loading_algorithm_is_ast_identical_except_explicit_adapter_boundaries(self):
        source = inspect.getsource(emotion_fusion.prepare_emotion_features)
        source = source.replace("adapter: EmotionDatasetAdapter, ", "")
        source = source.replace('    label_policy = label_policy_for(adapter.dataset)\n', '')
        source = source.replace('adapter.read_manifest(', 'read_feature_manifest(')
        source = source.replace('"adapter": adapter.identity(), ', '')
        source = source.replace('"dataset": adapter.dataset', '"dataset": "pmemo"')
        source = source.replace(', identities, audit, label_policy)', ', identities, audit)')
        candidate = ast.parse(source)
        reference = ast.parse(inspect.getsource(emotion_mid_fusion.prepare_emotion_features))
        for tree in (candidate, reference):
            tree.body[0].body.pop(0)  # Documentation is intentionally dataset-neutral.
        self.assertEqual(ast.dump(candidate), ast.dump(reference))

    def test_training_prevalence_is_target_specific_and_excludes_validation_and_test(self):
        features = self.features()
        features.annotations["7"] = dict(features.annotations["1"])
        features.splits["train"].append(replace(features.splits["train"][0], track_id="7"))
        valence, arousal = features.target("valence"), features.target("arousal")
        self.assertEqual(valence.mixture.prevalence.tolist(), [1 / 3])
        self.assertEqual(arousal.mixture.prevalence.tolist(), [2 / 3])
        self.assertEqual(valence.mixture.prevalence.dtype, np.float64)
        self.assertIs(valence.splits["train"][0].views["mixture"].segments,
                      arousal.splits["train"][0].views["mixture"].segments)

    def test_optional_checksums_preserve_scientific_identities(self):
        verified, trusted = self.features(), self.features(verify_checksums=False)
        for target in ("valence", "arousal"):
            first, second = verified.target(target), trusted.target(target)
            self.assertEqual(first.cohort_fingerprint, second.cohort_fingerprint)
            self.assertEqual(first.condition(SOURCES).data_fingerprint, second.condition(SOURCES).data_fingerprint)
            self.assertFalse(second.audit["cache_verification"]["feature_payload_checksums_verified"])

    def test_wrong_dataset_scales_and_one_class_splits_fail_without_cohort_filtering(self):
        for updates in ({"dataset": "pmemo"}, {"annotations": {"valence": 0.5, "arousal": 0.7}}):
            original = copy.deepcopy(self.rows[0])
            self.rows[0].update(updates)
            self.write_manifest()
            with self.assertRaises(ValueError):
                self.features()
            self.rows[0] = original
        self.rows[0]["annotations"]["valence"] = 5.0
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Both low/high"):
            self.features().target("valence")

    def test_corrupt_payload_and_unverified_alignment_and_audio_mismatch_are_rejected(self):
        payload = self.config.mert_dir / "mert_30s_guitar/1/segment_0.npy"
        original = payload.read_bytes()
        np.save(payload, np.full((1, 12, 768), np.nan, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "Changed or incomplete"):
            self.features()
        with self.assertRaisesRegex(ValueError, "finite float32"):
            self.features(verify_checksums=False)
        payload.write_bytes(original)
        path = self.config.mert_dir / "mert_30s_guitar/1/segments.json"
        metadata = json.loads(path.read_text())
        metadata["reference_mixture"] = {"path": "wrong"}
        write_json(path, metadata)
        with self.assertRaisesRegex(ValueError, "reference, settings or segment boundaries"):
            self.features(verify_checksums=False)

    def test_seven_conditions_five_seeds_both_targets_and_safe_complete_reuse(self):
        features = self.features()
        with self.tiny_models(), contextlib.redirect_stdout(io.StringIO()):
            result = controller.run_mid_fusion(self.config, features=features)
            before = {path: file_checksum(path) for path in self.config.output_dir.rglob('*') if path.is_file()}
            with patch.object(music2emo, "set_seed", side_effect=AssertionError("No retraining")):
                reused = controller.run_mid_fusion(self.config, features=features)
        self.assertEqual(len(list(self.config.output_dir.rglob("best.pt"))), 70)
        self.assertTrue(all(file_checksum(path) == sha for path, sha in before.items()))
        for target in ("valence", "arousal"):
            summary = result["targets"][target]
            self.assertEqual(summary["dataset"], "deam")
            self.assertEqual(len(summary["conditions"]), 7)
            self.assertEqual(len(summary["paired_differences"]), 3)
            self.assertEqual(summary["seeds"], [32, 64, 128, 256, 512])
            self.assertEqual(summary["conditions"], reused["targets"][target]["conditions"])
            for models in summary["runs"].values():
                for model in models.values():
                    self.assertEqual(model["protocol"]["label_policy"]["threshold"], 5.0)
                    self.assertEqual(model["protocol"]["target"], target)
            self.assertTrue((self.config.summary_dir / f"summary_mid_fusion_{target}.json").is_file())
            text = (self.config.summary_dir / f"summary_mid_fusion_{target}.txt").read_text()
            self.assertIn("original 1-9 score >= 5", text)
            self.assertIn("No score normalization", text)
            self.assertIn("mean +/- sample SD (ddof=1)", text)

    def test_same_labels_features_and_seed_produce_bit_identical_pmemo_and_deam_fits(self):
        config = replace(self.config, targets=("valence",), seeds=(32,), selected_run="mixture_only", epochs=2)
        previous = replace(self.fixture.config, targets=("valence",), seeds=(32,), selected_run="mixture_only", epochs=2)
        with self.tiny_models(), contextlib.redirect_stdout(io.StringIO()):
            a = controller.run_mid_fusion(config, features=self.features())["targets"]["valence"]["runs"]["mixture_only"][32]
            b = pmemo.run_mid_fusion(previous, features=self.fixture.features())["targets"]["valence"]["runs"]["mixture_only"][32]
        self.assertEqual(a["test"], b["test"])
        self.assertEqual(a["best_epoch"], b["best_epoch"])
        self.assertEqual(a["threshold"], b["threshold"])
        for split in ("validation", "test"):
            with np.load(config.output_dir / f"valence/mixture_only/seed_32/predictions_{split}.npz") as first, \
                    np.load(previous.output_dir / f"valence/mixture_only/seed_32/predictions_{split}.npz") as second:
                np.testing.assert_array_equal(first["logits"], second["logits"])

    def test_interrupted_fit_resume_is_exact_and_protocol_changes_are_rejected(self):
        config = replace(self.config, targets=("valence",), seeds=(32,), selected_run="mixture_only", epochs=2)
        features, original_fit = self.features(), music2emo.fit_model
        def interrupted(*args, **kwargs):
            def stop(epoch):
                raise RuntimeError("synthetic interruption")
            return original_fit(*args, **kwargs, epoch_callback=stop)
        with self.tiny_models(), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(music2emo, "fit_model", side_effect=interrupted), self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                controller.run_mid_fusion(config, features=features)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                controller.run_mid_fusion(replace(config, learning_rate=2e-4), features=features)
            resumed = controller.run_mid_fusion(config, features=features)
            fresh = controller.run_mid_fusion(replace(config, output_dir=self.root / "fresh"), features=features)
        self.assertEqual(resumed["targets"]["valence"]["conditions"], fresh["targets"]["valence"]["conditions"])

    def test_real_binary_architecture_fits_with_shared_encoder_and_single_output(self):
        config = replace(self.config, targets=("valence",), seeds=(32,), selected_run="mixture_guitar_drums_piano")
        with contextlib.redirect_stdout(io.StringIO()):
            result = controller.run_mid_fusion(config, features=self.features())
        fit = result["targets"]["valence"]["runs"][config.selected_run][32]
        self.assertEqual(fit["trainable_parameters"], controller.parameter_count(4))
        checkpoint = torch.load(fit["checkpoint"], map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["model_state"]["output_proj.2.weight"].shape, (1, 256))
        with np.load(config.output_dir / f"valence/{config.selected_run}/seed_32/predictions_test.npz") as saved:
            self.assertEqual(saved["logits"].shape, (2, 1))

    def test_validate_only_and_dry_run_do_not_fit_or_write(self):
        for changes in ({"validate_only": True}, {"dry_run": True}):
            with patch.object(music2emo, "fit_model", side_effect=AssertionError("No fitting")), \
                    patch.object(controller, "Music2EmoEmotionMidFusion", side_effect=AssertionError("No model")), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = controller.run_mid_fusion(replace(self.config, **changes))
            self.assertTrue(result[next(iter(changes))])
        self.assertFalse(self.config.output_dir.exists())
        self.assertFalse(self.config.summary_dir.exists())

    def test_yaml_matches_pmemo_optimization_and_native_rule_is_fixed(self):
        preset = default_config_path("experiments/deam/mid_fusion.yaml")
        defaults, provenance = load_configuration(preset, "deam_mid_fusion")
        previous, _ = load_configuration(default_config_path("experiments/pmemo/mid_fusion.yaml"), "emotion_mid_fusion")
        for key in ("epochs", "batch_size", "learning_rate", "early_stopping_patience", "early_stopping_min_delta",
                    "seeds", "device", "num_workers", "targets", "runs", "verify_checksums", "resume"):
            self.assertEqual(defaults[key], previous[key], key)
        self.assertEqual(len(provenance["config_files"]), 3)
        invalid = self.root / "invalid.yaml"
        for settings in ("task:\n  label_threshold: 0.5\n", "task:\n  score_scale: zero_one\n", "dataset: pmemo\n",
                         "model:\n  output_labels: 56\n", "runtime:\n  verify_checksums: 'false'\n"):
            invalid.write_text(f"training_profile: {preset}\n" + settings)
            with self.assertRaises(ConfigurationError):
                load_configuration(invalid, "deam_mid_fusion")

    def test_direct_and_unified_commands_resolve_deam_and_isolate_output_overrides(self):
        environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        result = subprocess.run([sys.executable, str(Path(train_mid.__file__)), "--dry-run", "--target", "arousal"],
            cwd=self.root, env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["dataset"], "deam")
        args = build_parser().parse_args(["train", "deam", "mid-fusion", "--target", "valence", "--seed", "64",
            "--run", "mixture_only", "--no-verify-checksums", "--output-dir", str(self.root / "isolated"), "--dry-run"])
        config = controller.config_from_args(args)
        self.assertEqual((config.dataset, config.targets, config.seeds), ("deam", ("valence",), (64,)))
        self.assertEqual(config.selected_run, "mixture_only")
        self.assertEqual(config.summary_dir, self.root / "isolated")
        self.assertFalse(config.verify_checksums)
        with patch.object(controller, "run_mid_fusion") as run:
            self.assertEqual(train_mid.main(["--dry-run"]), 0)
        self.assertEqual(run.call_args.args[0].dataset, "deam")


if __name__ == "__main__":
    unittest.main()

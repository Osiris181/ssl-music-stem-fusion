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
import torch

from stemfusion.configuration import ConfigurationError, default_config_path, load_configuration
from stemfusion.data.jamendo_reproduction import (
    JamendoReproductionDataset, encode_harmony, prepare_reproduction_data,
)
from stemfusion.experiments.mtg_jamendo.reproduction import (
    SEEDS, ReproductionConfig, _update_early_stopping, epoch_seed, fit_seed, format_summary, run_reproduction, summarize, weighted_bce,
)
from stemfusion.features.chords_keys import AUTHOR_COMMIT
from stemfusion.features.mert_segments import MODEL_NAME, MODEL_REVISION
from stemfusion.models.music2emo import Music2EmoClassifier
from stemfusion.training.reporting import finish_artifacts, write_json


class JamendoReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = ReproductionConfig(self.root / "manifest.jsonl", self.root / "mert", self.root / "harmony",
            self.root / "tags.npy", self.root / "metadata", self.root / "runs", self.root / "summaries",
            epochs=2, batch_size=2, device="cpu", num_workers=0)
        self.config.author_metadata.mkdir()
        self.names = [f"mood/theme---tag{i}" for i in range(56)]
        np.save(self.config.tag_list, np.array([f"other---{i}" for i in range(127)] + self.names))
        self.roots = {"N": 0, "C": 1, "D": 3}
        self.qualities = {"N": 0, "maj": 1, "min": 5}
        write_json(self.config.author_metadata / "chord_root.json", self.roots)
        write_json(self.config.author_metadata / "chord_attr.json", self.qualities)
        np.save(self.config.author_metadata / "probas_train.npy", np.full(56, 0.5, dtype=np.float64))
        self.rows = []
        for index in range(6):
            number = index + 1
            split = ("train", "validation", "test")[index // 2]
            row = {"track_id": f"track_{number:07d}", "split": split, "relative_path": f"{number:02d}/{number}.mp3",
                   "tags": self.names[index % 2::2], "paths": {"mixture": str(self.root / "absent-audio.mp3")}}
            self.rows.append(row)
            directory = self.config.mert_dir / "mert_30s" / f"{number:02d}/{number}"
            directory.mkdir(parents=True)
            for segment in range(2):
                value = np.zeros((1, 12, 768), dtype=np.float32)
                value[:, 5] = number / 100 + segment / 10
                value[:, 6] = number / 50 + segment / 10
                np.save(directory / f"segment_{segment}.npy", value)
            metadata = {"track_id": row["track_id"], "split": split, "source": "mixture",
                "settings": {"model_name": MODEL_NAME, "revision": MODEL_REVISION, "shape": [1, 12, 768],
                    "saved_hidden_states": list(range(1, 13)), "sample_rate": 24000, "segment_seconds": 30.0,
                    "temporal_pooling": "mean"}, "segments": [{"file": f"segment_{i}.npy"} for i in range(2)]}
            write_json(directory / "segments.json", metadata)
            finish_artifacts(directory, "fixture", ["segments.json", "segment_0.npy", "segment_1.npy"])
            harmony = self.config.harmony_dir / "mixture" / f"{number:02d}/{number}"
            harmony.mkdir(parents=True)
            (harmony / "normalized_chords.lab").write_text("0.000 1.000 C\n1.000 2.000 D:min\n")
            (harmony / "key.lab").write_text("None\n" if index == 0 else "D minor\n")
            write_json(harmony / "metadata.json", {"track_id": row["track_id"], "split": split, "source": "mixture",
                "settings": {"author_commit": AUTHOR_COMMIT, "midi_policy": "music2emotion_released", "normalization": "C_major_A_minor"},
                "extraction": {}})
            finish_artifacts(harmony, "fixture", ["normalized_chords.lab", "key.lab", "metadata.json"])
        self.config.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def tearDown(self):
        self.temporary.cleanup()

    def data(self):
        return prepare_reproduction_data(self.config.manifest, self.config.mert_dir, self.config.harmony_dir,
                                         self.config.tag_list, self.config.author_metadata)

    def test_verified_data_preserves_splits_layers_tags_and_author_key_fallback(self):
        data = self.data()
        self.assertEqual(data.audit["split_counts"], {"train": 2, "validation": 2, "test": 2})
        self.assertEqual(data.audit["segment_count"], 12)
        self.assertEqual(data.audit["unavailable_keys_mapped_to_major_as_author"], 1)
        first = data.splits["train"][0]
        self.assertEqual(first.segments.shape, (2, 1536))
        np.testing.assert_allclose(first.segments[:, 0], [0.01, 0.11])
        np.testing.assert_allclose(first.segments[:, 768], [0.02, 0.12])
        self.assertEqual(first.mode, 0)
        self.assertEqual(data.splits["train"][1].mode, 1)
        self.assertFalse((self.root / "absent-audio.mp3").exists())

    def test_sampling_matches_released_conditional_uniform_and_eval_mean(self):
        data = self.data()
        np.random.seed(32)
        start = np.random.randint(0, 2)
        end = np.random.randint(start + 1, 3)
        np.random.seed(32)
        train = JamendoReproductionDataset(data.splits["train"], training=True)[0]
        np.testing.assert_array_equal(train["x_mert"].numpy(), data.splits["train"][0].segments[start:end].mean(0))
        evaluation = JamendoReproductionDataset(data.splits["train"], training=False)[0]
        np.testing.assert_array_equal(evaluation["x_mert"].numpy(), data.splits["train"][0].segments.mean(0))

    def test_encoding_preserves_author_tokens_padding_truncation_and_none_mode(self):
        roots, qualities, mode = encode_harmony(["0 1 X", "1 2 C", "2 3 D:min"], "None", self.roots, self.qualities)
        np.testing.assert_array_equal(roots[:4], [0, 1, 3, 0])
        np.testing.assert_array_equal(qualities[:4], [0, 1, 5, 0])
        self.assertEqual((len(roots), mode), (100, 0))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            encode_harmony(["0 1 C:minmaj7"], "C major", self.roots, self.qualities)

    def test_corrupted_cache_or_wrong_prevalence_is_rejected_without_exclusions(self):
        directory = self.config.mert_dir / "mert_30s/01/1"
        (directory / "segment_0.npy").write_bytes(b"corrupted embedding")
        with self.assertRaisesRegex(ValueError, "Changed or incomplete"):
            self.data()

    def test_prevalence_must_match_train_labels_and_tag_order(self):
        np.save(self.config.author_metadata / "probas_train.npy", np.full(56, .25))
        with self.assertRaisesRegex(ValueError, "do not match"):
            self.data()

    def test_model_geometry_and_chord_parameters_are_trainable(self):
        model = Music2EmoClassifier()
        self.assertEqual(model.input_proj[0].in_features, 1545)
        self.assertEqual(model.input_proj[0].out_features, 512)
        self.assertEqual(model.output_proj[0].out_features, 256)
        self.assertEqual(len(model.chord_transformer.layers), 2)
        sample = JamendoReproductionDataset(self.data().splits["train"], training=False)[0]
        output = model({key: value.unsqueeze(0) for key, value in sample.items() if key.startswith("x_")})
        self.assertEqual(tuple(output.shape), (1, 56))
        output.sum().backward()
        self.assertGreater(float(model.chord_root_embedding.weight.grad.abs().sum()), 0)

    def test_weighted_bce_matches_author_formula_and_retains_float64_weights(self):
        prevalence = np.linspace(.1, .9, 56, dtype=np.float64)
        logits = torch.linspace(-1, 1, 112).reshape(2, 56)
        targets = torch.arange(112).reshape(2, 56).remainder(2).float()
        positive = torch.tensor(1.) / torch.from_numpy(prevalence)
        weight = torch.tensor(2.) / (torch.tensor(1.) + positive)
        expected = torch.nn.BCEWithLogitsLoss(pos_weight=positive, weight=weight)(logits, targets)
        criterion = weighted_bce(prevalence, "cpu")
        self.assertEqual(criterion.pos_weight.dtype, torch.float64)
        torch.testing.assert_close(criterion(logits, targets), expected, rtol=0, atol=0)

    def test_five_seed_matrix_saves_results_predictions_checkpoints_and_sample_sd(self):
        summary = run_reproduction(replace(self.config, epochs=1))
        self.assertEqual(summary["seeds"], list(SEEDS))
        self.assertEqual(len(summary["runs"]), 5)
        self.assertEqual(summary["budget"], "nonpaper_budget_override")
        values = summary["metrics"]["pr_auc_macro"]["values"]
        self.assertAlmostEqual(summary["metrics"]["pr_auc_macro"]["sample_sd"], float(np.std(values, ddof=1)))
        for seed in SEEDS:
            folder = self.config.output_dir / f"seed_{seed}"
            for name in ("best.pt", "last.pt", "result.json", "history.json", "predictions_test.npz", "predictions_validation.npz", "run_manifest.json"):
                self.assertTrue((folder / name).is_file())
        self.assertTrue((self.config.summary_dir / "summary_author_reproduction.txt").is_file())
        self.assertIn("sample SD", format_summary(summary))
        self.assertIn("nonpaper_budget_override", (self.config.summary_dir / "summary_author_reproduction.json").read_text())

    def test_completed_seed_reuses_without_model_initialization_and_rejects_changed_budget(self):
        data = self.data()
        result = fit_seed(self.config, data, 32)
        with patch("stemfusion.experiments.mtg_jamendo.reproduction.Music2EmoClassifier", side_effect=AssertionError("No retraining")):
            again = fit_seed(self.config, data, 32)
            self.assertEqual(result["test"], again["test"])
            with self.assertRaisesRegex(ValueError, "incompatible"):
                fit_seed(replace(self.config, epochs=3), data, 32)

    def test_exact_validation_loss_ties_keep_first_epoch_before_patience_is_exhausted(self):
        from stemfusion.experiments.mtg_jamendo import reproduction

        original = reproduction._evaluate

        def constant_loss(*args, **kwargs):
            if not kwargs.get("predictions"):
                return {"loss": 1.0}
            return original(*args, **kwargs)

        with patch.object(reproduction, "_evaluate", side_effect=constant_loss):
            result = fit_seed(self.config, self.data(), 32)
        self.assertEqual((result["best_epoch"], result["epochs_completed"]), (1, 2))
        self.assertEqual(result["early_stopping"]["wait_count"], 1)
        self.assertFalse(result["stopped_early"])

    def _validation_losses(self, values):
        from stemfusion.experiments.mtg_jamendo import reproduction

        original = reproduction._evaluate
        values = iter(values)

        def evaluate(*args, **kwargs):
            if not kwargs.get("predictions"):
                return {"loss": next(values)}
            return original(*args, **kwargs)

        return patch.object(reproduction, "_evaluate", side_effect=evaluate)

    def test_early_stopping_uses_ten_validation_checks_and_preserves_top_five_ties(self):
        config = replace(self.config, epochs=200)
        with self._validation_losses([1.] * 11):
            result = fit_seed(config, self.data(), 32)
        self.assertEqual((result["best_epoch"], result["epochs_completed"]), (1, 11))
        self.assertTrue(result["stopped_early"])
        self.assertEqual(result["stop_reason"], "early_stopping")
        self.assertEqual(result["early_stopping"], {"best_score": 1., "wait_count": 10, "stopped_epoch": 11})
        self.assertEqual([row["epoch"] for row in result["top_checkpoints"]], [1, 2, 3, 4, 5])
        folder = config.output_dir / "seed_32"
        self.assertEqual(len(list((folder / "checkpoints").glob("*.pt"))), 5)
        checkpoint = torch.load(folder / "last.pt", weights_only=True)
        self.assertEqual(checkpoint["early_stopping"], result["early_stopping"])
        self.assertIn("paper_200_epoch_cap", summarize([result])["budget"])
        self.assertIn("11 | 1 | early_stopping", format_summary(summarize([result])))

    def test_small_loss_decreases_update_best_checkpoint_without_resetting_patience(self):
        config = replace(self.config, epochs=20)
        losses = [1. - epoch * 1e-6 for epoch in range(11)]
        with self._validation_losses(losses):
            result = fit_seed(config, self.data(), 32)
        self.assertEqual((result["best_epoch"], result["epochs_completed"]), (11, 11))
        self.assertEqual(result["early_stopping"]["best_score"], 1.)
        self.assertEqual([row["epoch"] for row in result["top_checkpoints"]], [11, 10, 9, 8, 7])
        folder = config.output_dir / "seed_32"
        self.assertEqual(len(list((folder / "checkpoints").glob("*.pt"))), 5)
        self.assertEqual(torch.load(folder / "best.pt", weights_only=True)["epoch"], 11)

    def test_min_delta_is_strict_and_accumulated_improvement_resets_patience(self):
        state = {"best_score": float("inf"), "wait_count": 0, "stopped_epoch": None}
        for epoch, loss in enumerate([1., .875, .75], start=1):
            _update_early_stopping(state, loss, epoch, patience=10, min_delta=.125)
            if epoch == 2:
                self.assertEqual(state, {"best_score": 1., "wait_count": 1, "stopped_epoch": None})
        self.assertEqual(state, {"best_score": .75, "wait_count": 0, "stopped_epoch": None})

    def test_resume_restores_patience_and_finalizes_already_stopped_epoch_without_training(self):
        data = self.data()
        config = replace(self.config, epochs=20)
        with self._validation_losses([1.] * 11):
            uninterrupted = fit_seed(config, data, 32)
        restarted = replace(config, output_dir=self.root / "resumed-stopping")

        def cancel(epoch):
            if epoch in (5, 11):
                raise RuntimeError("synthetic interruption")

        with self._validation_losses([1.] * 5), self.assertRaisesRegex(RuntimeError, "interruption"):
            fit_seed(restarted, data, 32, epoch_callback=cancel)
        with self._validation_losses([1.] * 6), self.assertRaisesRegex(RuntimeError, "interruption"):
            fit_seed(restarted, data, 32, epoch_callback=cancel)
        callback = Mock(side_effect=AssertionError("Stopped run must not train another epoch"))
        resumed = fit_seed(restarted, data, 32, epoch_callback=callback)
        callback.assert_not_called()
        self.assertEqual(resumed["early_stopping"], uninterrupted["early_stopping"])
        self.assertEqual(resumed["test"], uninterrupted["test"])
        self.assertEqual(resumed["epochs_completed"], 11)
        self.assertEqual((restarted.output_dir / "seed_32/history.json").read_text(),
                         (config.output_dir / "seed_32/history.json").read_text())

    def test_ranked_checkpoint_corruption_is_rejected(self):
        data = self.data()

        def cancel(epoch):
            raise RuntimeError("synthetic interruption")

        with self.assertRaisesRegex(RuntimeError, "interruption"):
            fit_seed(self.config, data, 32, epoch_callback=cancel)
        (self.config.output_dir / "seed_32/checkpoints/epoch_0001.pt").write_bytes(b"corrupted checkpoint")
        with self.assertRaisesRegex(ValueError, "Ranked checkpoint"):
            fit_seed(self.config, data, 32)

    def test_summary_labels_overrides_and_rejects_mixed_stopping_protocols(self):
        config = replace(self.config, epochs=200, early_stopping_patience=0)
        with self._validation_losses([1., 1.]):
            result = fit_seed(config, self.data(), 32)
        summary = summarize([result])
        self.assertEqual(summary["budget"], "paper_200_epoch_cap")
        self.assertEqual(summary["early_stopping_rule"], "overridden")
        other = {**result, "seed": 64, "protocol": {**result["protocol"], "early_stopping": {
            **result["protocol"]["early_stopping"], "patience": 10}}}
        with self.assertRaisesRegex(ValueError, "identical training and stopping"):
            summarize([result, other])

    def test_repeatability_with_seeded_data_loader_workers(self):
        data = self.data()
        config = replace(self.config, epochs=1, num_workers=2)
        first = fit_seed(config, data, 64)
        second = fit_seed(replace(config, output_dir=self.root / "second-worker-run"), data, 64)
        self.assertEqual(first["test"], second["test"])

    def test_epoch_resume_is_bit_exact_to_uninterrupted_training(self):
        data = self.data()
        uninterrupted = fit_seed(self.config, data, 32)
        restarted_config = replace(self.config, output_dir=self.root / "resumed")

        def cancel(epoch):
            if epoch == 1:
                raise RuntimeError("synthetic interruption")

        with self.assertRaisesRegex(RuntimeError, "interruption"):
            fit_seed(restarted_config, data, 32, epoch_callback=cancel)
        resumed = fit_seed(restarted_config, data, 32)
        self.assertEqual(resumed["best_epoch"], uninterrupted["best_epoch"])
        self.assertEqual(resumed["test"], uninterrupted["test"])
        for name in ("history.json", "predictions_test.npz"):
            if name.endswith(".json"):
                self.assertEqual((self.config.output_dir / "seed_32" / name).read_text(), (restarted_config.output_dir / "seed_32" / name).read_text())
            else:
                first = np.load(self.config.output_dir / "seed_32" / name)
                second = np.load(restarted_config.output_dir / "seed_32" / name)
                np.testing.assert_array_equal(first["logits"], second["logits"])
        self.assertNotEqual(epoch_seed(32, 1), epoch_seed(32, 2))

    def test_yaml_defaults_and_direct_command_without_pythonpath(self):
        path = default_config_path("experiments/mtg_jamendo/reproduction.yaml")
        values, _ = load_configuration(path, "jamendo_reproduction")
        self.assertEqual(values["epochs"], 200)
        self.assertEqual(values["seeds"], list(SEEDS))
        self.assertEqual((values["batch_size"], values["learning_rate"]), (8, 1e-4))
        self.assertEqual((values["early_stopping_patience"], values["early_stopping_min_delta"]), (10, 1e-4))
        self.assertEqual(Path(values["output_dir"]).name, "author_reproduction_early_stopping")
        self.assertTrue(values["resume"])
        invalid = self.root / "bad.yaml"
        invalid.write_text(f"training_profile: {path}\noptimization:\n  early_stopping:\n    monitor: pr_auc\n")
        with self.assertRaises(ConfigurationError):
            load_configuration(invalid, "jamendo_reproduction")
        script = Path(__file__).resolve().parents[2] / "src/stemfusion/experiments/mtg_jamendo/train_reproduction.py"
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        result = subprocess.run([sys.executable, str(script), "--dry-run", "--seed", "64"], cwd=self.root,
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("seeds=[64]", result.stderr)
        self.assertIn("patience=10, min_delta=0.0001", result.stderr)
        self.assertFalse(self.config.output_dir.exists())

    def test_validate_only_does_not_train_or_write_outputs(self):
        with patch("stemfusion.experiments.mtg_jamendo.reproduction.fit_seed", side_effect=AssertionError("No training")):
            result = run_reproduction(replace(self.config, validate_only=True))
        self.assertTrue(result["validate_only"])
        self.assertFalse(self.config.output_dir.exists())


if __name__ == "__main__":
    unittest.main()

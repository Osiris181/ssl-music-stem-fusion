"""PMEmo CNN target semantics, shared fitting, matched controls and resume."""
from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from stemfusion.configuration import JAMENDO_MID_CONDITIONS
from stemfusion.data.emotion_cnn_features import JamendoCNNDataset, load_emotion_cnn_features
from stemfusion.data.emotion_mid_fusion import LABEL_POLICY
from stemfusion.experiments import _cnn_emotion as controller
from stemfusion.experiments.pmemo.train_mid_cnn import main
from stemfusion.features.musicnn import SOURCES, VERSION, CHECKPOINT_HASHES, SOURCE_COMMIT, save_cache
from stemfusion.models.cnn_emotion import CNNEmotionClassifier, parameter_count
from stemfusion.training.emotion import BinaryEmotionTask
from stemfusion.training.reporting import file_checksum

ROOT = Path(__file__).resolve().parents[2]


class PMEmoCNNTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "pmemo.jsonl"
        self.features = self.root / "features"
        self.rows = [
            {"dataset": "pmemo", "track_id": str(index + 1), "relative_path": f"{index + 1}.mp3",
             "split": ("train", "validation", "test")[index // 2], "duration_seconds": 3.,
             "annotations": {"valence": .5 if index % 2 else .25, "arousal": .25 if index % 2 else .75},
             "paths": {source: str(self.root / "absent_audio" / source / f"{index + 1}.mp3") for source in SOURCES}}
            for index in range(6)
        ]
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        for index, row in enumerate(self.rows):
            for source_index, source in enumerate(SOURCES):
                metadata = {key: row[key] for key in ("track_id", "split", "annotations", "relative_path")}
                metadata.update({"dataset": "pmemo", "source": source, "audio_path": row["paths"][source],
                                 "manifest_sha256": file_checksum(self.manifest), "extraction_version": VERSION,
                                 "window_frames": 187, "hop_frames": 187, "sample_rate": 16000,
                                 "tail_policy": "right_pad_log_mel_zero", "pooling": "unweighted_window_mean",
                                 "spectrogram_frames": 188, "windows": 2, "padded_frames": 186,
                                 "aligned_samples": 48000,
                                 "model": {"checkpoint": "MSD_musicnn", "checkpoint_sha256": CHECKPOINT_HASHES,
                                           "source_commit": SOURCE_COMMIT}})
                save_cache(self.features / row["split"] / source / f"{row['track_id']}.npz",
                           np.full(200, index / 10 + source_index / 6, np.float32), metadata)
        self.config = controller.EmotionCNNConfig(
            self.manifest, self.features, self.root / "runs", self.root / "reports",
            JAMENDO_MID_CONDITIONS, LABEL_POLICY, seeds=(32,), epochs=1, batch_size=2,
            device="cpu", num_workers=0,
        )

    def load(self):
        with contextlib.redirect_stderr(io.StringIO()):
            return load_emotion_cnn_features(self.manifest, self.features, dataset="pmemo", label_policy=LABEL_POLICY)

    def arguments(self, *extra):
        return ["--manifest", str(self.manifest), "--feature-dir", str(self.features),
                "--output-dir", str(self.root / "cli"), *extra]

    def test_native_midpoint_labels_training_prevalence_and_plain_model(self):
        features = self.load()
        valence, arousal = features.target("valence"), features.target("arousal")
        np.testing.assert_array_equal([row.labels[0] for row in valence.splits["train"]], [0, 1])
        np.testing.assert_array_equal([row.labels[0] for row in arousal.splits["train"]], [1, 0])
        np.testing.assert_array_equal(valence.prevalence, [.5])
        self.assertEqual(features.annotations["2"]["valence"], .5)
        self.assertIs(valence.splits["train"][0].views, arousal.splits["train"][0].views)
        self.assertNotEqual(valence.data_fingerprint, arousal.data_fingerprint)
        sample = JamendoCNNDataset(valence.splits["train"], sources=("mixture",), copies=4)[0]
        for vector in sample["x_mert"].reshape(4, 200):
            np.testing.assert_array_equal(vector.numpy(), valence.splits["train"][0].views["mixture"])
        self.assertTrue(all(sample[key].numel() == 0 for key in ("x_chord_root", "x_chord_attr", "x_key")))
        model = CNNEmotionClassifier(4)
        self.assertFalse(any(isinstance(layer, (torch.nn.Embedding, torch.nn.TransformerEncoder)) for layer in model.modules()))
        self.assertFalse(any(isinstance(layer, torch.nn.Dropout) and layer.p for layer in model.modules()))
        self.assertEqual(model({"x_mert": sample["x_mert"].unsqueeze(0)}).shape, (1, 1))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), parameter_count(4))
        with self.assertRaisesRegex(ValueError, "Both low/high"):
            features.annotations["2"]["valence"] = .25
            features.target("valence")

    def test_invalid_existing_caches_and_missing_views_remain_errors(self):
        path = self.features / "train/guitar/1.npz"
        with np.load(path, allow_pickle=False) as cache:
            metadata, vector = json.loads(str(cache["metadata"])), cache["embedding"]
        for changed, message in (
            ({"annotations": {"valence": .9, "arousal": .9}}, "provenance mismatch"),
            ({"runtime": {"gpu_memory_mb": 1}}, "Mixed CNN extraction"),
            ({"aligned_samples": 47999}, "do not align"),
        ):
            save_cache(path, vector, {**metadata, **changed})
            with self.assertRaisesRegex(ValueError, message):
                self.load()
        save_cache(path, vector.astype(np.float64), metadata)
        with self.assertRaisesRegex(ValueError, "finite float32"):
            self.load()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "IDs must exactly match"):
            self.load()

    def test_cli_inherits_existing_pmemo_recipe_and_writes_nothing_for_validation(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(self.arguments("--dry-run")), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["seeds"], [32, 64, 128, 256, 512])
        self.assertEqual(plan["targets"], ["valence", "arousal"])
        self.assertEqual(plan["planned_fits"], 70)
        self.assertEqual((plan["epochs"], plan["batch_size"], plan["learning_rate"]), (200, 8, 1e-4))
        self.assertEqual((plan["early_stopping_patience"], plan["early_stopping_min_delta"]), (10, 1e-4))
        self.assertEqual(plan["label_policy"], LABEL_POLICY)
        self.assertEqual(plan["hidden_dimensions"], [512, 256])
        self.assertEqual(plan["output_labels"], 1)
        self.assertFalse(plan["harmony_features"])
        for copies, name in ((4, "mixture_guitar_drums_piano"), (5, "mixture_guitar_drums_piano_bass"),
                             (6, "mixture_all_five_stems")):
            self.assertEqual(plan["conditions"][name]["trainable_parameters"],
                             plan["conditions"][f"repeated_mixture_{copies}"]["trainable_parameters"])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(self.arguments("--validate-only")), 0)
        self.assertFalse((self.root / "cli").exists())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(self.arguments("--dry-run", "--target", "valence", "--seed", "64", "--run", "mixture_only")), 0)
        subset = json.loads(output.getvalue())
        self.assertEqual((subset["targets"], subset["seeds"], subset["planned_fits"]), (["valence"], [64], 1))

    def test_real_fitter_fourteen_fits_validation_thresholds_and_verified_resume(self):
        with contextlib.redirect_stderr(io.StringIO()):
            result = controller.run_experiment(self.config)
        self.assertEqual(set(result["targets"]), {"valence", "arousal"})
        for target, summary in result["targets"].items():
            self.assertEqual(len(summary["conditions"]), 7)
            self.assertEqual(len(summary["paired_differences"]), 3)
            self.assertIn("MSD_musicnn", controller.format_summary(summary))
            for name, runs in summary["runs"].items():
                run = runs[32]
                self.assertEqual(run["protocol"]["selection"], "min_validation_weighted_bce")
                self.assertEqual(run["protocol"]["sampling"], "fixed_track_window_mean")
                self.assertEqual(run["protocol"]["task"], "binary_emotion_mid_fusion")
                self.assertFalse(run["protocol"]["harmony_features"])
                self.assertEqual(run["trainable_parameters"], summary["parameter_counts"][name])
                self.assertEqual(run["protocol"]["early_stopping"]["patience"], 10)
                folder = self.root / "runs" / target / name / "seed_32"
                with np.load(folder / "predictions_validation.npz", allow_pickle=False) as saved:
                    arrays = {key: saved[key] for key in ("targets", "probabilities")}
                self.assertEqual(run["threshold"], BinaryEmotionTask(target).select_threshold(arrays))
                with np.load(folder / "predictions_test.npz", allow_pickle=False) as saved:
                    self.assertEqual(float(saved["threshold"]), run["threshold"])
            self.assertTrue(controller.summary_path(self.config, target).with_suffix(".txt").is_file())
        paths = list((self.root / "runs").rglob("best.pt"))
        self.assertEqual(len(paths), 14)
        before = {path: (file_checksum(path), path.stat().st_mtime_ns) for path in paths}
        with patch("torch.optim.Adam", side_effect=AssertionError("Compatible runs must not retrain")), \
             contextlib.redirect_stderr(io.StringIO()):
            repeated = controller.run_experiment(self.config)
        for target in result["targets"]:
            self.assertEqual(result["targets"][target]["conditions"], repeated["targets"][target]["conditions"])
        self.assertEqual(before, {path: (file_checksum(path), path.stat().st_mtime_ns) for path in before})
        with self.assertRaisesRegex(ValueError, "incompatible settings"), contextlib.redirect_stderr(io.StringIO()):
            controller.run_experiment(replace(self.config, epochs=2))

    def test_all_five_outer_seeds_schedule_separate_target_models(self):
        def fit(config, data, seed, **kwargs):
            target = kwargs["task"].target
            self.assertEqual(config.targets, (target,))
            self.assertEqual(config.output_dir.parts[-2], target)
            return {"seed": seed, "data_fingerprint": data.data_fingerprint,
                    "trainable_parameters": parameter_count(kwargs["protocol_updates"]["inputs"]),
                    "test": {name: .5 for name in ("accuracy", "precision_macro", "recall_macro",
                                                  "f1_macro", "roc_auc_macro", "pr_auc_macro")}}
        with patch.object(controller.music2emo, "fit_model", side_effect=fit) as fitter, \
             contextlib.redirect_stderr(io.StringIO()):
            result = controller.run_experiment(replace(self.config, seeds=controller.SEEDS))
        self.assertEqual(fitter.call_count, 70)
        for target, summary in result["targets"].items():
            self.assertEqual(summary["seeds"], list(controller.SEEDS))
            self.assertTrue(all(set(runs) == set(controller.SEEDS) for runs in summary["runs"].values()))
            self.assertTrue(all(value["data_fingerprint"] == summary["runs"]["mixture_only"][32]["data_fingerprint"]
                                for runs in summary["runs"].values() for value in runs.values()))

    def test_direct_training_entry_works_outside_repository(self):
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        script = ROOT / "src/stemfusion/experiments/pmemo/train_mid_cnn.py"
        result = subprocess.run([sys.executable, str(script), "--dry-run"], cwd=self.root,
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["planned_fits"], 70)


if __name__ == "__main__":
    unittest.main()

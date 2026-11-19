"""Native DEAM scores use the unchanged shared PMEmo CNN fitting workflow."""
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
from stemfusion.experiments import _cnn_emotion as controller
from stemfusion.experiments.deam import train_mid_cnn as entry
from stemfusion.experiments.pmemo.train_mid_cnn import main as pmemo_main
from stemfusion.features.musicnn import SOURCES, VERSION, CHECKPOINT_HASHES, SOURCE_COMMIT, save_cache
from stemfusion.models.cnn_emotion import CNNEmotionClassifier, parameter_count
from stemfusion.training.emotion import BinaryEmotionTask
from stemfusion.training.reporting import file_checksum

ROOT = Path(__file__).resolve().parents[2]
LABEL_POLICY = {"annotation_scale": "original_1_9", "threshold": 5.0,
                "positive_comparison": ">=", "negative_class": "low", "positive_class": "high"}


class DEAMCNNTrainingTests(unittest.TestCase):
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
        self.manifest = self.root / "deam.jsonl"
        self.features = self.root / "features"
        self.rows = []
        for split_index, split in enumerate(("train", "validation", "test")):
            valence = (1., 4.999, 5., 9.) if split == "train" else (5., 5., 9., 4.)
            arousal = (9., 5., 4.999, 1.) if split == "train" else (4., 4., 5., 1.)
            for index in range(4):
                track_id = str(split_index * 4 + index + 1)
                self.rows.append({"dataset": "deam", "track_id": track_id,
                                  "relative_path": f"{track_id}.mp3", "split": split,
                                  "annotations": {"valence": valence[index], "arousal": arousal[index]},
                                  "paths": {source: str(self.root / "absent_audio" / source / f"{track_id}.mp3")
                                            for source in SOURCES}})
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        for index, row in enumerate(self.rows):
            for source_index, source in enumerate(SOURCES):
                metadata = {key: row[key] for key in ("track_id", "split", "annotations", "relative_path")}
                metadata.update(dataset="deam", source=source, audio_path=row["paths"][source],
                                manifest_sha256=file_checksum(self.manifest), extraction_version=VERSION,
                                window_frames=187, hop_frames=187, sample_rate=16000,
                                tail_policy="right_pad_log_mel_zero", pooling="unweighted_window_mean",
                                spectrogram_frames=188, windows=2, padded_frames=186, aligned_samples=48000,
                                model={"checkpoint": "MSD_musicnn", "checkpoint_sha256": CHECKPOINT_HASHES,
                                       "source_commit": SOURCE_COMMIT})
                save_cache(self.features / row["split"] / source / f"{row['track_id']}.npz",
                           np.full(200, index / 10 + source_index / 6, np.float32), metadata)
        self.config = controller.EmotionCNNConfig(
            self.manifest, self.features, self.root / "runs", self.root / "reports",
            JAMENDO_MID_CONDITIONS, LABEL_POLICY, dataset="deam", seeds=(32,), epochs=1,
            batch_size=4, device="cpu", num_workers=0,
        )

    def load(self):
        with contextlib.redirect_stderr(io.StringIO()):
            return load_emotion_cnn_features(self.manifest, self.features, dataset="deam", label_policy=LABEL_POLICY)

    def arguments(self, *extra):
        return ["--manifest", str(self.manifest), "--feature-dir", str(self.features),
                "--output-dir", str(self.root / "cli"), *extra]

    def plan(self, main, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["--dry-run", *args]), 0)
        return json.loads(output.getvalue())

    def test_native_five_boundary_scores_training_prevalence_and_matched_inputs(self):
        before = self.manifest.read_bytes()
        features = self.load()
        valence, arousal = features.target("valence"), features.target("arousal")
        np.testing.assert_array_equal([row.labels[0] for row in valence.splits["train"]], [0, 0, 1, 1])
        np.testing.assert_array_equal([row.labels[0] for row in arousal.splits["train"]], [1, 1, 0, 0])
        np.testing.assert_array_equal(valence.prevalence, [.5])
        np.testing.assert_array_equal(arousal.prevalence, [.5])
        self.assertEqual(valence.audit["class_counts"]["validation"], {"low": 1, "high": 3})
        self.assertEqual(arousal.audit["class_counts"]["validation"], {"low": 3, "high": 1})
        self.assertEqual(features.annotations["3"]["valence"], 5.)
        self.assertEqual(features.annotations["2"]["valence"], 4.999)
        self.assertEqual(before, self.manifest.read_bytes())
        self.assertIs(valence.splits["train"][0].views, arousal.splits["train"][0].views)
        self.assertNotEqual(valence.data_fingerprint, arousal.data_fingerprint)
        for count in (4, 5, 6):
            item = JamendoCNNDataset(valence.splits["train"], sources=("mixture",), copies=count)[0]
            for vector in item["x_mert"].reshape(count, 200):
                np.testing.assert_array_equal(vector.numpy(), valence.splits["train"][0].views["mixture"])
            self.assertTrue(all(item[key].numel() == 0 for key in ("x_chord_root", "x_chord_attr", "x_key")))
            model = CNNEmotionClassifier(count)
            self.assertEqual(model({"x_mert": item["x_mert"].unsqueeze(0)}).shape, (1, 1))
            self.assertEqual(sum(value.numel() for value in model.parameters()), parameter_count(count))

    def test_config_uses_the_same_pmemo_recipe_except_native_labels_and_paths(self):
        plan = self.plan(entry.main)
        reference = self.plan(pmemo_main)
        for key in ("targets", "seeds", "planned_fits", "conditions", "epochs", "batch_size", "learning_rate",
                    "early_stopping_patience", "early_stopping_min_delta", "checkpoint", "threshold",
                    "hidden_dimensions", "output_labels", "dropout", "harmony_features"):
            self.assertEqual(plan[key], reference[key], key)
        self.assertEqual(plan["planned_fits"], 70)
        self.assertEqual(plan["seeds"], [32, 64, 128, 256, 512])
        self.assertEqual(plan["label_policy"], LABEL_POLICY)
        self.assertEqual(plan["feature_dir"], str(ROOT / "data/features/deam/msd_musicnn"))
        self.assertEqual(plan["output_dir"], str(ROOT / "outputs/deam/cnn/runs"))
        self.assertEqual(plan["summary_dir"], str(ROOT / "outputs/deam/cnn"))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(entry.main(self.arguments("--validate-only")), 0)
        self.assertFalse((self.root / "cli").exists())
        subset = self.plan(entry.main, "--target", "valence", "--seed", "64", "--run", "mixture_only")
        self.assertEqual((subset["targets"], subset["seeds"], subset["planned_fits"]), (["valence"], [64], 1))

    def test_real_fourteen_fits_thresholds_summaries_and_verified_resume(self):
        with contextlib.redirect_stderr(io.StringIO()):
            result = controller.run_experiment(self.config)
        self.assertEqual(set(result["targets"]), {"valence", "arousal"})
        for target, summary in result["targets"].items():
            self.assertEqual(len(summary["conditions"]), 7)
            self.assertEqual(len(summary["paired_differences"]), 3)
            self.assertEqual(summary["label_policy"], LABEL_POLICY)
            self.assertIn("DEAM: binary", controller.format_summary(summary))
            for name, runs in summary["runs"].items():
                run = runs[32]
                self.assertEqual(run["protocol"]["selection"], "min_validation_weighted_bce")
                self.assertEqual(run["protocol"]["sampling"], "fixed_track_window_mean")
                self.assertEqual(run["protocol"]["label_policy"], LABEL_POLICY)
                self.assertFalse(run["protocol"]["harmony_features"])
                self.assertEqual(run["trainable_parameters"], summary["parameter_counts"][name])
                folder = self.root / "runs" / target / name / "seed_32"
                with np.load(folder / "predictions_validation.npz", allow_pickle=False) as saved:
                    arrays = {key: saved[key] for key in ("targets", "probabilities")}
                self.assertEqual(run["threshold"], BinaryEmotionTask(target).select_threshold(arrays))
                with np.load(folder / "predictions_test.npz", allow_pickle=False) as saved:
                    self.assertEqual(float(saved["threshold"]), run["threshold"])
            self.assertTrue(controller.summary_path(self.config, target).is_file())
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

    def test_five_seeds_schedule_separate_targets_on_one_cohort(self):
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
        for summary in result["targets"].values():
            self.assertEqual(summary["seeds"], list(controller.SEEDS))
            self.assertTrue(all(set(runs) == set(controller.SEEDS) for runs in summary["runs"].values()))
            self.assertTrue(all(value["data_fingerprint"] == summary["runs"]["mixture_only"][32]["data_fingerprint"]
                                for runs in summary["runs"].values() for value in runs.values()))

    def test_missing_or_incompatible_cache_is_not_silently_accepted(self):
        path = self.features / "train/guitar/1.npz"
        with np.load(path, allow_pickle=False) as cache:
            metadata, vector = json.loads(str(cache["metadata"])), cache["embedding"]
        save_cache(path, vector, {**metadata, "annotations": {"valence": .5, "arousal": .5}})
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            self.load()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "IDs must exactly match"):
            self.load()

    def test_direct_command_works_without_pythonpath_or_audio(self):
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        result = subprocess.run([sys.executable, str(Path(entry.__file__)), "--dry-run"], cwd=self.root,
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["planned_fits"], 70)
        self.assertEqual(plan["label_policy"], LABEL_POLICY)


if __name__ == "__main__":
    unittest.main()

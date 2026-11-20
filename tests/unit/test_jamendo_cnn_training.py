"""Plain CNN Jamendo inputs and integration with the existing multilabel fitter."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from stemfusion.configuration import JAMENDO_MID_CONDITIONS, JAMENDO_LATE_CONDITIONS
from stemfusion.data.jamendo_cnn_features import JamendoCNNDataset, load_jamendo_cnn_features
from stemfusion.experiments._cnn_mood import JamendoCNNConfig, main, run_experiment, format_summary
from stemfusion.experiments import _cnn_mood_late as late
from stemfusion.features.musicnn import SOURCES, VERSION, CHECKPOINT_HASHES, SOURCE_COMMIT, save_cache
from stemfusion.models.cnn_mood import CNNMoodClassifier
from stemfusion.training import music2emo
from stemfusion.training.reporting import file_checksum, fingerprint
from stemfusion.training.late_fusion import branch_seed


class JamendoCNNTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.jsonl"
        self.feature_dir = self.root / "features"
        self.tag_list = self.root / "tags.npy"
        self.prevalence = self.root / "prevalence.npy"
        self.tags = [f"mood/theme---tag_{i}" for i in range(56)]
        np.save(self.tag_list, np.asarray([f"other---{i}" for i in range(127)] + self.tags))
        np.save(self.prevalence, np.full(56, .5, dtype=np.float64))
        self.rows = [
            {"track_id": f"track_{i:07d}", "split": ("train", "validation", "test")[(i - 1) // 2],
             "relative_path": f"{i:02d}/{i}.mp3", "tags": self.tags[(i - 1) % 2::2],
             "paths": {source: str(self.root / "absent_audio" / source / f"{i}.mp3") for source in SOURCES}}
            for i in range(1, 7)
        ]
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        checksum = file_checksum(self.manifest)
        for index, row in enumerate(self.rows):
            for source_index, source in enumerate(SOURCES):
                metadata = {key: row[key] for key in ("track_id", "split", "tags", "relative_path")}
                metadata.update({"dataset": "mtg_jamendo", "source": source,
                                 "audio_path": row["paths"][source], "manifest_sha256": checksum,
                                 "extraction_version": VERSION, "window_frames": 187, "hop_frames": 187,
                                 "sample_rate": 16000, "tail_policy": "right_pad_log_mel_zero",
                                 "pooling": "unweighted_window_mean", "spectrogram_frames": 188,
                                 "windows": 2, "padded_frames": 186, "aligned_samples": 48000,
                                 "model": {"checkpoint": "MSD_musicnn", "checkpoint_sha256": CHECKPOINT_HASHES,
                                           "source_commit": SOURCE_COMMIT}})
                save_cache(self.feature_dir / row["split"] / source / f"{row['track_id']}.npz",
                           np.full(200, index / 10 + source_index / 6, dtype=np.float32), metadata)
        self.config = JamendoCNNConfig(self.manifest, self.feature_dir, self.tag_list, self.prevalence,
            self.root / "runs", self.root / "reports", JAMENDO_MID_CONDITIONS,
            seeds=(32,), epochs=1, batch_size=2, device="cpu", num_workers=0)

    def load_data(self):
        with contextlib.redirect_stderr(io.StringIO()):
            return load_jamendo_cnn_features(self.manifest, self.feature_dir, self.tag_list, self.prevalence)

    def arguments(self, *extra):
        return ["--manifest", str(self.manifest), "--feature-dir", str(self.feature_dir),
                "--tag-list", str(self.tag_list), "--prevalence", str(self.prevalence),
                "--output-dir", str(self.root / "cli"), *extra]

    def test_labels_fixed_vectors_and_absence_of_harmony_parameters(self):
        data = self.load_data()
        self.assertEqual(data.tag_names, tuple(self.tags))
        self.assertEqual(data.audit["split_counts"], {"train": 2, "validation": 2, "test": 2})
        np.testing.assert_array_equal(data.splits["train"][0].labels, np.arange(56) % 2 == 0)
        dataset = JamendoCNNDataset(data.splits["train"], sources=("mixture",), copies=4)
        sample = dataset[0]
        for copy in sample["x_mert"].reshape(4, 200):
            np.testing.assert_array_equal(copy.numpy(), data.splits["train"][0].views["mixture"])
        for key in ("x_chord_root", "x_chord_attr", "x_key"):
            self.assertEqual(sample[key].numel(), 0)
        model = CNNMoodClassifier(4)
        self.assertFalse(any(isinstance(layer, (torch.nn.Embedding, torch.nn.TransformerEncoder))
                             for layer in model.modules()))
        self.assertFalse(any(isinstance(layer, torch.nn.Dropout) and layer.p for layer in model.modules()))
        with torch.inference_mode():
            logits = model({"x_mert": sample["x_mert"].unsqueeze(0)})
        self.assertEqual(logits.shape, (1, 56))

    def test_rejects_wrong_targets_prevalence_and_mixed_runtime(self):
        np.save(self.prevalence, np.full(56, .6, dtype=np.float64))
        with self.assertRaisesRegex(ValueError, "weights do not match"):
            self.load_data()
        np.save(self.prevalence, np.full(56, .5, dtype=np.float64))
        path = self.feature_dir / "train/guitar/track_0000001.npz"
        with np.load(path, allow_pickle=False) as cache:
            vector, metadata = cache["embedding"], json.loads(str(cache["metadata"]))
        save_cache(path, vector, {**metadata, "runtime": {"gpu_memory_mb": 999}})
        with self.assertRaisesRegex(ValueError, "Mixed CNN extraction"):
            self.load_data()
        save_cache(path, vector, {**metadata, "tags": self.tags[1::2]})
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            self.load_data()

    def test_cli_inherits_jamendo_settings_and_validation_writes_nothing(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(self.arguments("--dry-run")), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["seeds"], [32, 64, 128, 256, 512])
        self.assertEqual(plan["hidden_dimensions"], [512, 256])
        self.assertEqual(plan["epochs"], 200)
        self.assertEqual(plan["batch_size"], 8)
        self.assertEqual(plan["learning_rate"], 1e-4)
        self.assertEqual(plan["early_stopping_patience"], 10)
        self.assertFalse(plan["harmony_features"])
        for count, name in [(4, "mixture_guitar_drums_piano"), (5, "mixture_guitar_drums_piano_bass"),
                            (6, "mixture_all_five_stems")]:
            self.assertEqual(plan["conditions"][f"repeated_mixture_{count}"]["trainable_parameters"],
                             plan["conditions"][name]["trainable_parameters"])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(self.arguments("--validate-only")), 0)
        self.assertFalse((self.root / "cli").exists())

    def test_shared_fitter_all_conditions_checkpoints_thresholds_and_resume(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                result = run_experiment(self.config)
            self.assertEqual(len(result["conditions"]), 7)
            self.assertEqual(len(result["paired_differences"]), 3)
            self.assertNotIn("chord Transformer", format_summary(result))
            for name, runs in result["runs"].items():
                run = runs[32]
                self.assertEqual(run["protocol"]["selection"], "min_validation_weighted_bce")
                self.assertEqual(run["protocol"]["sampling"], "fixed_track_window_mean")
                self.assertEqual(run["protocol"]["early_stopping"]["patience"], 10)
                self.assertEqual(run["trainable_parameters"], result["parameter_counts"][name])
                with np.load(self.root / "runs" / name / "seed_32/predictions_validation.npz", allow_pickle=False) as saved:
                    arrays = {key: saved[key] for key in ("targets", "probabilities")}
                    self.assertEqual(run["threshold"], music2emo.select_threshold(arrays))
            self.assertTrue((self.root / "reports/summary_mid_fusion_seeds_32.txt").is_file())
            checkpoints = {path: path.stat().st_mtime_ns for path in (self.root / "runs").rglob("best.pt")}
            self.assertEqual(len(checkpoints), 7)
            with contextlib.redirect_stderr(io.StringIO()):
                repeated = run_experiment(self.config)
            self.assertEqual(result["conditions"], repeated["conditions"])
            self.assertEqual(checkpoints, {path: path.stat().st_mtime_ns for path in checkpoints})
        finally:
            torch.set_num_threads(threads)

    def late_config(self, **updates):
        return late.JamendoCNNLateConfig(
            **{**vars(self.config), "conditions": JAMENDO_LATE_CONDITIONS,
               "output_dir": self.root / "late", "summary_dir": self.root / "late_reports", **updates},
            reference_output_dir=self.config.output_dir)

    def fit_reference(self):
        with contextlib.redirect_stderr(io.StringIO()):
            return run_experiment(replace(self.config, selected_run="mixture_only"))

    def test_late_cli_inherits_mid_training_settings_and_position_paired_schedule(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = late.main(self.arguments("--reference-output-dir", str(self.config.output_dir), "--dry-run"))
        self.assertEqual(status, 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["seeds"], [32, 64, 128, 256, 512])
        self.assertEqual(plan["hidden_dimensions"], [512, 256])
        self.assertEqual((plan["epochs"], plan["batch_size"], plan["learning_rate"]), (200, 8, 1e-4))
        self.assertEqual((plan["early_stopping_patience"], plan["early_stopping_min_delta"]), (10, 1e-4))
        self.assertFalse(plan["harmony_features"])
        self.assertEqual(plan["dropout"], 0.)
        self.assertEqual(plan["branch_trainable_parameters"], 248632)
        self.assertEqual(plan["branches_per_outer_seed"], 11)
        self.assertEqual(plan["new_branch_fits_per_seed"], 10)
        self.assertEqual(plan["maximum_new_branch_fits"], 50)
        self.assertEqual(plan["report_conditions_per_seed"], 12)
        self.assertEqual(plan["aggregation"], "sum_logits_then_sigmoid")
        schedule = {item["branch_id"]: item for item in plan["branch_schedule"]["32"]}
        for index, source in enumerate(SOURCES[1:], start=1):
            self.assertEqual(schedule[source]["actual_training_seed"], branch_seed(32, index))
            self.assertEqual(schedule[source]["actual_training_seed"], schedule[f"mixture_{index + 1}"]["actual_training_seed"])
        self.assertFalse((self.root / "cli").exists())

    def test_late_missing_reference_and_overlapping_roots_fail_without_fitting(self):
        config = self.late_config()
        with patch.object(late.music2emo, "fit_model", side_effect=AssertionError("Must not fit")), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "Completed CNN mixture-only reference required"):
                late.run_experiment(config)
            with self.assertRaisesRegex(ValueError, "non-nested roots"):
                late.run_experiment(replace(config, output_dir=config.reference_output_dir / "late"))
        self.assertFalse(config.output_dir.exists())

    def test_late_subsets_request_only_their_required_new_branches(self):
        for name, fits in (("mixture_only", 0), ("guitar_only", 1),
                           ("mixture_ensemble_4", 3), ("mixture_guitar_drums_piano", 3)):
            with patch.object(late.mid, "load_jamendo_cnn_features", side_effect=AssertionError("Dry run must not load features")):
                plan = late.run_experiment(self.late_config(selected_run=name, dry_run=True))
            self.assertEqual(plan["new_branch_fits_per_seed"], fits)
            self.assertEqual(plan["report_conditions_per_seed"], 1)
        self.assertFalse((self.root / "late").exists())

    def test_late_full_run_reuses_m1_and_shared_fitter_thresholds_and_resume(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            self.fit_reference()
            config = self.late_config()
            reference_files = {path: (file_checksum(path), path.stat().st_mtime_ns)
                               for path in self.config.output_dir.rglob("*") if path.is_file()}
            with patch.object(late.music2emo, "fit_model", wraps=music2emo.fit_model) as fitting, \
                 contextlib.redirect_stderr(io.StringIO()):
                result = late.run_experiment(config)
            self.assertEqual(fitting.call_count, 10)
            self.assertEqual(len(result["conditions"]), 12)
            self.assertEqual(len(result["paired_differences"]), 3)
            self.assertEqual(len(result["standalone_conditions"]), 5)
            self.assertEqual(len(list(config.output_dir.rglob("best.pt"))), 10)
            self.assertFalse((config.output_dir / "branches/mixture_1").exists())
            branches = result["branches"][32]
            self.assertEqual(branches["mixture_1"]["output_directory"], str(self.config.output_dir / "mixture_only/seed_32"))
            for name, branch in branches.items():
                protocol = branch["protocol"]
                self.assertEqual(protocol["hidden_dimensions"], [512, 256])
                self.assertEqual(protocol["head_dropout"], 0.)
                self.assertFalse(protocol["harmony_features"])
                self.assertEqual(protocol["sampling"], "fixed_track_window_mean")
                self.assertEqual(protocol["selection"], "min_validation_weighted_bce")
                self.assertEqual(branch["trainable_parameters"], 248632)
                self.assertEqual(branch["actual_training_seed"], branch_seed(32, branch["index"]))
                if name != "mixture_1":
                    self.assertEqual(protocol["early_stopping"]["patience"], 10)
                with np.load(Path(branch["output_directory"]) / "predictions_validation.npz", allow_pickle=False) as saved:
                    self.assertEqual(branch["threshold"], music2emo.select_threshold(saved))
            for name, runs in result["runs"].items():
                run = runs[32]
                if JAMENDO_LATE_CONDITIONS[name]["kind"] in {"reference", "stem_only"}:
                    self.assertEqual(run["additional_training_jobs"], 0)
                    continue
                directory = Path(run["output_directory"])
                with np.load(directory / "predictions_validation.npz", allow_pickle=False) as saved:
                    self.assertEqual(run["threshold"], music2emo.select_threshold(saved))
                summed = np.zeros((2, 56), dtype=np.float32)
                for member in run["members"]:
                    with np.load(Path(member["output_directory"]) / "predictions_test.npz", allow_pickle=False) as saved:
                        summed += saved["logits"]
                with np.load(directory / "predictions_test.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["logits"], summed)
                    np.testing.assert_allclose(saved["probabilities"], torch.from_numpy(summed).sigmoid().numpy())
                    np.testing.assert_array_equal(saved["predictions"], saved["probabilities"] >= run["threshold"])
            report = self.root / "late_reports/summary_late_fusion_seeds_32.txt"
            self.assertTrue(report.is_file())
            self.assertIn("guitar_only", report.read_text())
            self.assertIn("sum raw logits", report.read_text())
            times = {path: path.stat().st_mtime_ns for path in config.output_dir.rglob("best.pt")}
            with patch("torch.optim.Adam", side_effect=AssertionError("Resume must not fit")), \
                 contextlib.redirect_stderr(io.StringIO()):
                repeated = late.run_experiment(config)
            self.assertEqual(result["conditions"], repeated["conditions"])
            self.assertEqual(times, {path: path.stat().st_mtime_ns for path in times})
            with contextlib.redirect_stderr(io.StringIO()):
                late.run_experiment(replace(config, overwrite=True, resume=False))
            self.assertEqual(reference_files, {path: (file_checksum(path), path.stat().st_mtime_ns)
                                               for path in reference_files})
            with self.assertRaisesRegex(ValueError, "incompatible batch_size"), \
                 contextlib.redirect_stderr(io.StringIO()):
                late.run_experiment(replace(config, output_dir=self.root / "different", batch_size=3))
            self.assertFalse((self.root / "different").exists())
        finally:
            torch.set_num_threads(threads)

    def test_late_reference_validation_preserves_mid_identity_and_rejects_changed_checkpoints(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            reference = self.fit_reference()["runs"]["mixture_only"][32]
            with contextlib.redirect_stderr(io.StringIO()):
                result = late.run_experiment(self.late_config(validate_only=True))
            self.assertEqual(result["reference_seeds_verified"], [32])
            self.assertFalse((self.root / "late").exists())
            with patch.object(late.music2emo, "fit_model", side_effect=AssertionError("Must preflight all references")), \
                 self.assertRaisesRegex(ValueError, "Completed CNN mixture-only reference required"), \
                 contextlib.redirect_stderr(io.StringIO()):
                late.run_experiment(self.late_config(seeds=(32, 64)))
            self.assertFalse((self.root / "late").exists())
            from stemfusion.experiments._cnn_mood import implementation_identity
            self.assertEqual(reference["implementation_sha256"], implementation_identity())
            signature = fingerprint({"seed": 32, "protocol": reference["protocol"],
                                     "data": reference["data_fingerprint"], "implementation": implementation_identity()})
            self.assertEqual(signature, reference["signature"])
            Path(reference["checkpoint"]).write_bytes(b"damaged synthetic checkpoint")
            with self.assertRaisesRegex(ValueError, "artifact is missing or changed"), \
                 contextlib.redirect_stderr(io.StringIO()):
                late.run_experiment(self.late_config(validate_only=True))
            self.assertFalse((self.root / "late").exists())
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()

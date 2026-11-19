"""FMA CNN cache/label alignment and shared trainer integration."""
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

from stemfusion.configuration import load_configuration
from stemfusion.data.features import FeatureDataError
from stemfusion.data.fma_cnn_features import load_fma_cnn_features
from stemfusion.experiments._genre import GenreExperimentConfig, run_experiment, format_results
from stemfusion.experiments.fma_medium.train_mid_cnn import main
from stemfusion.experiments.fma_medium.train_late_cnn import main as late_main
from stemfusion.features.musicnn import SOURCES, VERSION, CHECKPOINT_HASHES, SOURCE_COMMIT, save_cache
from stemfusion.training.reporting import file_checksum
from stemfusion.training.late_fusion import branch_seed, train_branch
from stemfusion.training.trainer import TrainingSettings

ROOT = Path(__file__).resolve().parents[2]


class FMACNNTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "fma.jsonl"
        self.features = self.root / "features"
        rows = [
            {"track_id": f"{index * 32 + example * 16 + label + 1:06d}", "genre": f"genre_{label}", "label": label,
             "split": split, "paths": {source: "/unused/audio" for source in (*SOURCES, "other")}}
            for index, split in enumerate(("train", "validation", "test"))
            for example in range(2) for label in range(16)
        ]
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        checksum = file_checksum(self.manifest)
        for row in rows:
            for index, source in enumerate(SOURCES):
                metadata = {key: row[key] for key in ("track_id", "split", "label")}
                metadata.update({"dataset": "fma_medium", "source": source,
                                 "manifest_sha256": checksum, "extraction_version": VERSION,
                                 "window_frames": 187, "hop_frames": 187, "sample_rate": 16000,
                                 "tail_policy": "right_pad_log_mel_zero", "pooling": "unweighted_window_mean",
                                 "spectrogram_frames": 188, "windows": 2, "padded_frames": 186,
                                 "aligned_samples": 48000,
                                 "model": {"checkpoint": "MSD_musicnn", "checkpoint_sha256": CHECKPOINT_HASHES,
                                           "source_commit": SOURCE_COMMIT}})
                save_cache(self.features / row["split"] / source / f"{row['track_id']}.npz",
                           np.full(200, row["label"] / 16 + index / 6, dtype=np.float32), metadata)

    def arguments(self, *extra):
        return ["--manifest", str(self.manifest), "--feature-dir", str(self.features),
                "--output-dir", str(self.root / "runs"), *extra]

    def test_ids_labels_and_silent_vectors(self):
        data = load_fma_cnn_features(self.manifest, self.features)
        self.assertEqual(data.number_of_classes, 16)
        split = data.splits["train"]
        self.assertEqual(split.track_ids[0], "000001")
        np.testing.assert_array_equal(split.labels, np.tile(np.arange(16), 2))
        self.assertEqual(split.features["mixture"].shape, (32, 200))
        self.assertTrue(np.all(split.features["mixture"][0] == 0))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(self.arguments("--validate-only")), 0)
        self.assertFalse((self.root / "runs").exists())

    def test_wrong_label_still_fails(self):
        path = self.features / "train/guitar/000001.npz"
        with np.load(path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata"]))
            vector = cache["embedding"]
        save_cache(path, vector, {**metadata, "label": 15})
        with self.assertRaisesRegex(FeatureDataError, "provenance mismatch"):
            load_fma_cnn_features(self.manifest, self.features)

    def test_missing_views_skip_whole_tracks_from_every_source(self):
        (self.features / "train/guitar/000001.npz").unlink()
        (self.features / "train/drums/000002.npz").unlink()
        with self.assertLogs("stemfusion.data.fma_cnn_features", level="WARNING"):
            data = load_fma_cnn_features(self.manifest, self.features)
        self.assertEqual(data.audit["manifest_split_counts"], {"train": 32, "validation": 32, "test": 32})
        self.assertEqual(data.audit["split_counts"], {"train": 30, "validation": 32, "test": 32})
        self.assertEqual(data.audit["skipped_missing_features"], [
            {"track_id": "000001", "split": "train", "missing_sources": ["guitar"]},
            {"track_id": "000002", "split": "train", "missing_sources": ["drums"]},
        ])
        self.assertNotIn("000001", data.splits["train"].track_ids)
        self.assertNotIn("000002", data.splits["train"].track_ids)
        for source in SOURCES:
            self.assertEqual(data.splits["train"].features[source].shape, (30, 200))
        self.assertTrue((self.features / "train/mixture/000001.npz").is_file())
        self.assertTrue((self.features / "train/mixture/000002.npz").is_file())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(self.arguments("--validate-only")), 0)
        self.assertFalse((self.root / "runs").exists())

    def test_unknown_ids_empty_splits_and_lost_classes_remain_errors(self):
        extra = self.features / "train/guitar/999999.npz"
        original = self.features / "train/guitar/000001.npz"
        extra.write_bytes(original.read_bytes())
        with self.assertRaisesRegex(FeatureDataError, "outside the train manifest"):
            load_fma_cnn_features(self.manifest, self.features)
        extra.unlink()
        original.unlink()
        (self.features / "train/guitar/000017.npz").unlink()
        with self.assertRaisesRegex(FeatureDataError, "retain all 16 classes"):
            load_fma_cnn_features(self.manifest, self.features)
        for file in (self.features / "train/guitar").glob("*.npz"):
            file.unlink()
        with self.assertRaisesRegex(FeatureDataError, "No complete FMA CNN tracks"):
            load_fma_cnn_features(self.manifest, self.features)

    def test_plan_inherits_protocol_and_matches_parameters(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(self.arguments("--dry-run")), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["seeds"], [32, 64, 128, 256, 512])
        self.assertEqual(plan["settings"], {"hidden_dimensions": [1024, 512], "dropout": .2,
                                         "learning_rate": .0001, "batch_size": 128,
                                         "epochs": 20, "device": "auto", "num_workers": 0})
        for count, name in [(4, "mixture_guitar_drums_piano"), (5, "mixture_guitar_drums_piano_bass"),
                            (6, "mixture_all_five_stems")]:
            control = plan["conditions"][f"repeated_mixture_{count}"]
            self.assertEqual(control["input_matching"]["input_dimension"], 200 * count)
            self.assertEqual(control["trainable_parameters"], plan["conditions"][name]["trainable_parameters"])
        self.assertFalse((self.root / "runs").exists())

    def test_all_seven_conditions_share_complete_tracks_fit_report_and_resume(self):
        (self.features / "train/drums/000001.npz").unlink()
        _, provenance = load_configuration(ROOT / "configs/experiments/fma_medium/mid_fusion.yaml", "mid_fusion")
        config = GenreExperimentConfig(
            manifest_path=self.manifest, feature_directory=self.features, output_directory=self.root / "runs",
            summary_directory=self.root / "reports", family="mid_fusion", dataset="fma_medium",
            feature_backend="msd_musicnn", dimension=200, number_of_classes=16,
            conditions=provenance["experiment_metadata"]["runs"], seeds=(32,),
            settings=TrainingSettings(epochs=1, device="cpu"),
        )
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            result = run_experiment(config)
            self.assertEqual(len(result["conditions"]), 7)
            self.assertEqual(len(result["paired_differences"]), 3)
            self.assertEqual(result["split_counts"], {"train": 31, "validation": 32, "test": 32})
            for conditions in result["runs"].values():
                self.assertEqual(conditions[32]["split_examples"], {"train": 31, "validation": 32, "test": 32})
            self.assertIn("FMA-Medium", format_results(result))
            self.assertIn("MSD_musicnn", format_results(result))
            self.assertNotIn("MusicFM-MSD", format_results(result))
            self.assertTrue((self.root / "reports/summary_mid_fusion_seeds_32.txt").is_file())
            checkpoints = {path: path.stat().st_mtime_ns for path in (self.root / "runs").rglob("checkpoint.pt")}
            self.assertEqual(len(checkpoints), 7)
            repeated = run_experiment(config)
            self.assertEqual(result["conditions"], repeated["conditions"])
            self.assertEqual(checkpoints, {path: path.stat().st_mtime_ns for path in checkpoints})
        finally:
            torch.set_num_threads(threads)

    def late_config(self, **updates):
        _, provenance = load_configuration(ROOT / "configs/experiments/fma_medium/late_fusion.yaml", "late_fusion")
        config = GenreExperimentConfig(
            manifest_path=self.manifest, feature_directory=self.features,
            output_directory=self.root / "late", reference_output_directory=self.root / "mid",
            summary_directory=self.root / "reports", family="late_fusion", dataset="fma_medium",
            feature_backend="msd_musicnn", dimension=200, number_of_classes=16,
            conditions=provenance["experiment_metadata"]["runs"], seeds=(32,),
            settings=TrainingSettings(epochs=1, device="cpu"),
        )
        return replace(config, **updates)

    def fit_mixture_reference(self, config, data):
        mid = replace(config, family="mid_fusion", reference_output_directory=None,
                      output_directory=config.reference_output_directory,
                      summary_directory=self.root / "mid_reports",
                      conditions={"mixture_only": {"kind": "reference", "sources": ["mixture"]}})
        return run_experiment(mid, data=data)

    def test_late_plan_inherits_musicfm_protocol_and_isolates_outputs(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = late_main(self.arguments("--dry-run"))
        self.assertEqual(status, 0)
        plan = json.loads(output.getvalue())
        defaults, _ = load_configuration(ROOT / "configs/experiments/fma_medium/late_fusion.yaml", "late_fusion")
        self.assertEqual(plan["settings"], {key: defaults[key] for key in
                         ("hidden_dimensions", "dropout", "learning_rate", "batch_size", "epochs", "device", "num_workers")})
        self.assertEqual(plan["seeds"], [32, 64, 128, 256, 512])
        self.assertEqual(len(plan["conditions"]), 6)
        self.assertEqual(plan["maximum_new_branch_fits"], 50)
        self.assertEqual(plan["new_branch_fits_per_seed"], 10)
        self.assertEqual(plan["reference_mode"], "read_only_completed_cnn_mixture_1")
        for count in (4, 5, 6):
            self.assertEqual(plan["conditions"][f"mixture_ensemble_{count}"]["trainable_parameters"], count * 738832)
        self.assertFalse((self.root / "runs").exists())

    def test_late_missing_reference_and_overlapping_roots_fail_without_fitting(self):
        config = self.late_config()
        data = load_fma_cnn_features(self.manifest, self.features)
        with patch("stemfusion.experiments._genre.train_branch", side_effect=AssertionError("Must not fit")):
            with self.assertRaisesRegex(ValueError, "Completed CNN mixture-only reference required"):
                run_experiment(config, data=data)
            with self.assertRaisesRegex(ValueError, "non-nested roots"):
                run_experiment(replace(config, output_directory=config.reference_output_directory / "late"), data=data)
        self.assertFalse(config.output_directory.exists())

    def test_late_reuses_m1_fits_ten_branches_and_reports_retained_cohort(self):
        (self.features / "train/drums/000001.npz").unlink()
        data = load_fma_cnn_features(self.manifest, self.features)
        config = self.late_config()
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            self.fit_mixture_reference(config, data)
            reference_files = {path: (file_checksum(path), path.stat().st_mtime_ns)
                               for path in config.reference_output_directory.rglob("*") if path.is_file()}
            with patch("stemfusion.experiments._genre.train_branch", wraps=train_branch) as fitting:
                result = run_experiment(config, data=data)
            self.assertEqual(fitting.call_count, 10)
            scheduled = {(call.kwargs["source"], call.kwargs["index"]) for call in fitting.call_args_list}
            self.assertEqual(scheduled, {("mixture", index) for index in range(1, 6)} |
                             {(source, index) for index, source in enumerate(SOURCES) if index})
            self.assertEqual(len(list(config.output_directory.rglob("checkpoint.pt"))), 10)
            self.assertFalse((config.output_directory / "branches/seed_32/mixture_1").exists())
            self.assertEqual(len(result["conditions"]), 6)
            self.assertEqual(len(result["standalone_stems"]), 5)
            self.assertEqual(len(result["paired_differences"]), 3)
            self.assertEqual(result["split_counts"], {"train": 31, "validation": 32, "test": 32})
            self.assertEqual(result["mixture_reference_runs"][32]["seed"], 32)
            text = (self.root / "reports/summary_late_fusion_seeds_32.txt").read_text()
            self.assertIn("MSD_musicnn", text)
            self.assertIn("Supporting analysis: individual stem classifiers", text)
            stem_path = self.root / "reports/summary_stem_only_seeds_32.json"
            self.assertEqual(json.loads(stem_path.read_text())["split_counts"]["train"], 31)
            for source, index in (("mixture", 1), ("guitar", 1)):
                name = "mixture_2" if source == "mixture" else source
                metrics = json.loads((config.output_directory / f"branches/seed_32/{name}/metrics.json").read_text())
                self.assertEqual(metrics["seed"], branch_seed(32, index))
                self.assertEqual(metrics["model"]["input_dimension"], 200)
                self.assertEqual(metrics["model"]["hidden_dimensions"], [1024, 512])
            ensemble = result["runs"]["mixture_all_five_stems"][32]
            summed = np.zeros((32, 16), dtype=np.float32)
            for source in SOURCES:
                root = config.reference_output_directory if source == "mixture" else config.output_directory
                name = "mixture_1" if source == "mixture" else source
                with np.load(root / f"branches/seed_32/{name}/predictions.npz", allow_pickle=False) as saved:
                    summed += saved["logits"]
            with np.load(Path(ensemble["output_directory"]) / "predictions.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["logits"], summed)
            times = {path: path.stat().st_mtime_ns for path in config.output_directory.rglob("checkpoint.pt")}
            repeated = run_experiment(config, data=data)
            self.assertEqual(result["conditions"], repeated["conditions"])
            self.assertEqual(times, {path: path.stat().st_mtime_ns for path in times})
            run_experiment(replace(config, overwrite=True), data=data)
            self.assertEqual(reference_files, {path: (file_checksum(path), path.stat().st_mtime_ns)
                                               for path in reference_files})
            with self.assertRaisesRegex(ValueError, "incompatible settings"):
                run_experiment(replace(config, output_directory=self.root / "different",
                                       settings=replace(config.settings, dropout=0.)), data=data)
            self.assertFalse((self.root / "different").exists())
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()

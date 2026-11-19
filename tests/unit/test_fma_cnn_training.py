"""FMA CNN cache/label alignment and shared trainer integration."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from stemfusion.configuration import load_configuration
from stemfusion.data.features import FeatureDataError
from stemfusion.data.fma_cnn_features import load_fma_cnn_features
from stemfusion.experiments._genre import GenreExperimentConfig, run_experiment, format_results
from stemfusion.experiments.fma_medium.train_mid_cnn import main
from stemfusion.features.musicnn import SOURCES, VERSION, CHECKPOINT_HASHES, SOURCE_COMMIT, save_cache
from stemfusion.training.reporting import file_checksum
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


if __name__ == "__main__":
    unittest.main()

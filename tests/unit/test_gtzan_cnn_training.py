"""CNN cache contract and CPU integration with the unchanged genre trainer."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from stemfusion.data.cnn_features import load_gtzan_cnn_features
from stemfusion.data.features import FeatureDataError
from stemfusion.data.gtzan import LABEL_MAPPING
from stemfusion.features.musicnn import SOURCES, VERSION, CHECKPOINT_HASHES, SOURCE_COMMIT, save_cache
from stemfusion.experiments.gtzan.train_mid_cnn import main
from stemfusion.configuration import load_configuration
from stemfusion.experiments._genre import GenreExperimentConfig, run_experiment, format_results
from stemfusion.training.trainer import TrainingSettings
from stemfusion.training.reporting import file_checksum

ROOT = Path(__file__).resolve().parents[2]


class GTZANCNNTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.jsonl"
        self.features = self.root / "features"
        rows = []
        for index, split in enumerate(("train", "validation", "test")):
            for genre, label in LABEL_MAPPING.items():
                rows.append({"track_id": f"{genre}.{index:05d}", "genre": genre, "label": label,
                             "split": split, "paths": {source: "/unused/audio" for source in SOURCES}})
        self.manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        for row in rows:
            for source in SOURCES:
                metadata = {key: row[key] for key in ("track_id", "split", "label")}
                metadata.update({"dataset": "gtzan", "source": source, "manifest_sha256": file_checksum(self.manifest),
                                 "extraction_version": VERSION, "window_frames": 187, "hop_frames": 187,
                                 "sample_rate": 16000, "tail_policy": "right_pad_log_mel_zero",
                                 "pooling": "unweighted_window_mean", "spectrogram_frames": 188,
                                 "windows": 2, "padded_frames": 186,
                                 "model": {"checkpoint": "MSD_musicnn", "checkpoint_sha256": CHECKPOINT_HASHES,
                                           "source_commit": SOURCE_COMMIT}})
                save_cache(self.features / row["split"] / source / f"{row['track_id']}.npz",
                           np.full(200, row["label"] / 10, dtype=np.float32), metadata)
        _, provenance = load_configuration(ROOT / "configs/experiments/gtzan/mid_fusion.yaml", "mid_fusion")
        self.config = GenreExperimentConfig(
            manifest_path=self.manifest, feature_directory=self.features, output_directory=self.root / "runs",
            summary_directory=self.root / "summaries", family="mid_fusion", dataset="gtzan",
            feature_backend="msd_musicnn", dimension=200, number_of_classes=10,
            conditions=provenance["experiment_metadata"]["runs"], seeds=(32,),
            settings=TrainingSettings(epochs=1, device="cpu"),
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_cache_alignment_and_corruption(self):
        data = load_gtzan_cnn_features(self.manifest, self.features)
        self.assertEqual(data.splits["train"].features["mixture"].shape, (10, 200))
        self.assertEqual(data.audit["split_counts"], {"train": 10, "validation": 10, "test": 10})
        path = self.features / "train/mixture/blues.00000.npz"
        with np.load(path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata"]))
        save_cache(path, np.full(200, np.nan, dtype=np.float32), metadata)
        with self.assertRaises(FeatureDataError):
            load_gtzan_cnn_features(self.manifest, self.features)

    def test_all_conditions_fit_report_and_resume(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            result = run_experiment(self.config)
            self.assertEqual(len(result["conditions"]), 7)
            self.assertEqual(len(result["paired_differences"]), 3)
            text = format_results(result)
            self.assertIn("MSD_musicnn", text)
            self.assertNotIn("MusicFM-MSD", text)
            self.assertTrue((self.root / "summaries/summary_mid_fusion_seeds_32.txt").is_file())
            checkpoints = {path: path.stat().st_mtime_ns for path in (self.root / "runs").rglob("checkpoint.pt")}
            self.assertEqual(len(checkpoints), 7)
            repeated = run_experiment(self.config)
            self.assertEqual(result["conditions"], repeated["conditions"])
            self.assertEqual(checkpoints, {path: path.stat().st_mtime_ns for path in checkpoints})
        finally:
            torch.set_num_threads(threads)

    def test_cli_inherits_default_settings_and_redirects_paths(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(["--manifest", str(self.manifest), "--feature-dir", str(self.features),
                           "--output-dir", str(self.root / "custom"), "--dry-run"])
        self.assertEqual(status, 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["seeds"], [32, 64, 128, 256, 512])
        self.assertEqual(plan["settings"]["hidden_dimensions"], [1024, 512])
        self.assertEqual(plan["settings"]["epochs"], 20)
        self.assertEqual(plan["summary_directory"], str(self.root / "custom"))
        self.assertEqual(plan["conditions"]["repeated_mixture_4"]["input_matching"]["input_dimension"], 800)
        self.assertFalse((self.root / "custom").exists())


if __name__ == "__main__":
    unittest.main()

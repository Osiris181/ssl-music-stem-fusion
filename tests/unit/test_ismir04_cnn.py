"""ISMIR04 CNN entry points, manifest mapping, and shared fitting integration."""
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
from stemfusion.data.ismir04 import LABEL_MAPPING
from stemfusion.data.ismir04_cnn_features import load_ismir04_cnn_features
from stemfusion.experiments._cnn_extraction import read_records
from stemfusion.experiments._genre import GenreExperimentConfig, run_experiment, format_results
from stemfusion.experiments.ismir04.train_mid_cnn import main
from stemfusion.features.musicnn import SOURCES, VERSION, CHECKPOINT_HASHES, SOURCE_COMMIT, save_cache
from stemfusion.training.reporting import file_checksum
from stemfusion.training.trainer import TrainingSettings

ROOT = Path(__file__).resolve().parents[2]


class ISMIR04CNNTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "ismir04.jsonl"
        self.features = self.root / "features"
        self.rows = [
            {"track_id": f"{split}__{genre}__synthetic", "genre": genre, "label": label,
             "split": split, "paths": {source: "/unused/audio" for source in SOURCES}}
            for split in ("train", "validation", "test") for genre, label in LABEL_MAPPING.items()
        ]
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        for row in self.rows:
            for source in SOURCES:
                metadata = {key: row[key] for key in ("track_id", "split", "label")}
                metadata.update({"dataset": "ismir04", "source": source,
                                 "manifest_sha256": file_checksum(self.manifest), "extraction_version": VERSION,
                                 "window_frames": 187, "hop_frames": 187, "sample_rate": 16000,
                                 "tail_policy": "right_pad_log_mel_zero", "pooling": "unweighted_window_mean",
                                 "spectrogram_frames": 188, "windows": 2, "padded_frames": 186,
                                 "aligned_samples": 48000,
                                 "model": {"checkpoint": "MSD_musicnn", "checkpoint_sha256": CHECKPOINT_HASHES,
                                           "source_commit": SOURCE_COMMIT}})
                save_cache(self.features / row["split"] / source / f"{row['track_id']}.npz",
                           np.full(200, row["label"] / 6, dtype=np.float32), metadata)

    def arguments(self, *extra):
        return ["--manifest", str(self.manifest), "--feature-dir", str(self.features),
                "--output-dir", str(self.root / "runs"), *extra]

    def test_manifest_and_cohort_validation(self):
        records = read_records(self.manifest, dataset="ismir04")
        self.assertEqual(len(records), 18)
        data = load_ismir04_cnn_features(self.manifest, self.features)
        self.assertEqual(data.label_mapping, LABEL_MAPPING)
        self.assertEqual(data.splits["train"].features["mixture"].shape, (6, 200))
        self.assertEqual(data.audit["split_counts"], {"train": 6, "validation": 6, "test": 6})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(self.arguments("--validate-only")), 0)
        self.assertFalse((self.root / "runs").exists())
        self.assertEqual(json.loads(output.getvalue())["number_of_classes"], 6)

    def test_rejects_wrong_dataset_and_misaligned_windows(self):
        path = self.features / "train/guitar/train__classical__synthetic.npz"
        with np.load(path, allow_pickle=False) as cache:
            vector = cache["embedding"]
            metadata = json.loads(str(cache["metadata"]))
        save_cache(path, vector, {**metadata, "dataset": "gtzan"})
        with self.assertRaisesRegex(FeatureDataError, "provenance mismatch"):
            load_ismir04_cnn_features(self.manifest, self.features)
        save_cache(path, vector, {**metadata, "aligned_samples": 47999})
        with self.assertRaisesRegex(FeatureDataError, "do not align"):
            load_ismir04_cnn_features(self.manifest, self.features)

    def test_plan_inherits_genre_protocol(self):
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

    def test_seven_conditions_use_shared_trainer_and_resume(self):
        _, provenance = load_configuration(ROOT / "configs/experiments/ismir04/mid_fusion.yaml", "mid_fusion")
        config = GenreExperimentConfig(
            manifest_path=self.manifest, feature_directory=self.features, output_directory=self.root / "runs",
            summary_directory=self.root / "reports", family="mid_fusion", dataset="ismir04",
            feature_backend="msd_musicnn", dimension=200, number_of_classes=6,
            conditions=provenance["experiment_metadata"]["runs"], seeds=(32,),
            settings=TrainingSettings(epochs=1, device="cpu"),
        )
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            result = run_experiment(config)
            self.assertEqual(len(result["conditions"]), 7)
            self.assertEqual(len(result["paired_differences"]), 3)
            text = format_results(result)
            self.assertIn("ISMIR04", text)
            self.assertIn("MSD_musicnn", text)
            self.assertNotIn("MusicFM-MSD", text)
            self.assertTrue((self.root / "reports/summary_mid_fusion_seeds_32.txt").is_file())
            paths = list((self.root / "runs").rglob("checkpoint.pt"))
            self.assertEqual(len(paths), 7)
            times = {path: path.stat().st_mtime_ns for path in paths}
            again = run_experiment(config)
            self.assertEqual(result["conditions"], again["conditions"])
            self.assertEqual(times, {path: path.stat().st_mtime_ns for path in paths})
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()

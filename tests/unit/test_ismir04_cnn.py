"""ISMIR04 CNN entry points, manifest mapping, and shared fitting integration."""
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
from stemfusion.data.ismir04 import LABEL_MAPPING
from stemfusion.data.ismir04_cnn_features import load_ismir04_cnn_features
from stemfusion.experiments._cnn_extraction import read_records
from stemfusion.experiments._genre import GenreExperimentConfig, run_experiment, format_results
from stemfusion.experiments.ismir04.train_mid_cnn import main
from stemfusion.experiments.ismir04.train_late_cnn import main as late_main
from stemfusion.features.musicnn import SOURCES, VERSION, CHECKPOINT_HASHES, SOURCE_COMMIT, save_cache
from stemfusion.training.reporting import file_checksum
from stemfusion.training.trainer import TrainingSettings
from stemfusion.training.late_fusion import branch_seed, train_branch

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

    def late_arguments(self, *extra):
        return self.arguments("--reference-output-dir", str(self.root / "mid"), *extra)

    def test_late_plan_matches_other_genre_protocols_and_workload(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(late_main(self.late_arguments("--dry-run")), 0)
        plan = json.loads(output.getvalue())
        settings, provenance = load_configuration(ROOT / "configs/experiments/ismir04/late_fusion.yaml", "late_fusion")
        for dataset in ("fma_medium", "gtzan"):
            other, _ = load_configuration(ROOT / f"configs/experiments/{dataset}/late_fusion.yaml", "late_fusion")
            for key in ("hidden_dimensions", "dropout", "learning_rate", "batch_size", "epochs", "device", "num_workers"):
                self.assertEqual(plan["settings"][key], settings[key])
                self.assertEqual(settings[key], other[key])
        self.assertEqual(plan["seeds"], [32, 64, 128, 256, 512])
        self.assertEqual(set(plan["conditions"]), set(provenance["experiment_metadata"]["runs"]))
        self.assertEqual(plan["new_branch_fits_per_seed"], 10)
        self.assertEqual(plan["maximum_new_branch_fits"], 50)
        self.assertEqual(plan["reference_mode"], "read_only_completed_cnn_mixture_1")
        for count in (4, 5, 6):
            self.assertEqual(plan["conditions"][f"mixture_ensemble_{count}"]["trainable_parameters"], count * 733702)
        self.assertFalse((self.root / "runs").exists())

    def test_late_validation_requires_m1_without_fitting_or_writing(self):
        with patch("stemfusion.experiments._genre.train_branch", side_effect=AssertionError("Must not fit")), \
             contextlib.redirect_stdout(io.StringIO()), self.assertLogs(level="ERROR") as messages:
            status = late_main(self.late_arguments("--seed", "32", "--epochs", "1", "--device", "cpu", "--validate-only"))
        self.assertEqual(status, 2)
        self.assertIn("Completed CNN mixture-only reference required", " ".join(messages.output))
        self.assertFalse((self.root / "runs").exists())
        self.assertFalse((self.root / "mid").exists())

    def test_late_cli_reuses_m1_and_reports_all_ensembles_and_stems(self):
        _, provenance = load_configuration(ROOT / "configs/experiments/ismir04/mid_fusion.yaml", "mid_fusion")
        config = GenreExperimentConfig(
            manifest_path=self.manifest, feature_directory=self.features, output_directory=self.root / "mid",
            summary_directory=self.root / "mid_reports", family="mid_fusion", dataset="ismir04",
            feature_backend="msd_musicnn", dimension=200, number_of_classes=6,
            conditions=provenance["experiment_metadata"]["runs"], seeds=(32,),
            settings=TrainingSettings(epochs=1, device="cpu"),
        )
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            run_experiment(replace(config, selected_run="mixture_only"))
            reference_files = {path: (file_checksum(path), path.stat().st_mtime_ns)
                               for path in config.output_directory.rglob("*") if path.is_file()}
            args = self.late_arguments("--seed", "32", "--epochs", "1", "--device", "cpu")
            output = io.StringIO()
            with contextlib.redirect_stdout(output), \
                 patch("stemfusion.experiments._genre.train_branch", wraps=train_branch) as fitting:
                self.assertEqual(late_main(args), 0)
            self.assertEqual(fitting.call_count, 10)
            self.assertEqual({(call.kwargs["source"], call.kwargs["index"]) for call in fitting.call_args_list},
                             {("mixture", index) for index in range(1, 6)} |
                             {(source, index) for index, source in enumerate(SOURCES) if index})
            root = self.root / "runs"
            summary = json.loads((root / "summary_late_fusion_seeds_32.json").read_text())
            self.assertEqual(summary["number_of_classes"], 6)
            self.assertEqual(summary["split_counts"], {"train": 6, "validation": 6, "test": 6})
            self.assertEqual(len(summary["conditions"]), 6)
            self.assertEqual(len(summary["standalone_stems"]), 5)
            self.assertEqual(len(summary["paired_differences"]), 3)
            self.assertIn("MSD_musicnn", output.getvalue())
            self.assertIn("Supporting analysis: individual stem classifiers", output.getvalue())
            self.assertTrue((root / "summary_stem_only_seeds_32.txt").is_file())
            self.assertEqual(len(list(root.rglob("checkpoint.pt"))), 10)
            self.assertFalse((root / "branches/seed_32/mixture_1").exists())
            summed = np.zeros((6, 6), dtype=np.float32)
            for index, source in enumerate(SOURCES):
                branch_root = config.output_directory if source == "mixture" else root
                name = "mixture_1" if source == "mixture" else source
                directory = branch_root / f"branches/seed_32/{name}"
                metrics = json.loads((directory / "metrics.json").read_text())
                self.assertEqual(metrics["seed"], branch_seed(32, index))
                self.assertEqual(metrics["model"]["input_dimension"], 200)
                self.assertEqual(metrics["model"]["number_of_classes"], 6)
                self.assertEqual(metrics["model"]["hidden_dimensions"], [1024, 512])
                with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
                    summed += saved["logits"]
            ensemble = Path(summary["runs"]["mixture_all_five_stems"]["32"]["output_directory"])
            with np.load(ensemble / "predictions.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["logits"], summed)
            times = {path: path.stat().st_mtime_ns for path in root.rglob("checkpoint.pt")}
            with contextlib.redirect_stdout(io.StringIO()), \
                 patch("torch.optim.Adam", side_effect=AssertionError("Resume must not fit")):
                self.assertEqual(late_main(args), 0)
            self.assertEqual(times, {path: path.stat().st_mtime_ns for path in times})
            self.assertEqual(reference_files, {path: (file_checksum(path), path.stat().st_mtime_ns)
                                               for path in reference_files})
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()

"""CNN cache contract and CPU integration with the unchanged genre trainer."""
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

from stemfusion.data.cnn_features import load_gtzan_cnn_features
from stemfusion.data.features import FeatureDataError
from stemfusion.data.gtzan import LABEL_MAPPING
from stemfusion.features.musicnn import SOURCES, VERSION, CHECKPOINT_HASHES, SOURCE_COMMIT, save_cache
from stemfusion.experiments.gtzan.train_mid_cnn import main
from stemfusion.experiments.gtzan.train_late_cnn import main as late_main
from stemfusion.configuration import load_configuration
from stemfusion.experiments._genre import GenreExperimentConfig, run_experiment, format_results
from stemfusion.training.trainer import TrainingSettings
from stemfusion.training.reporting import file_checksum
from stemfusion.training.late_fusion import branch_seed, train_branch

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

    def late_arguments(self, *extra):
        return ["--manifest", str(self.manifest), "--feature-dir", str(self.features),
                "--reference-output-dir", str(self.config.output_directory),
                "--output-dir", str(self.root / "late"), *extra]

    def test_late_cli_inherits_same_protocol_and_workload_as_fma(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = late_main(self.late_arguments("--dry-run"))
        self.assertEqual(status, 0)
        plan = json.loads(output.getvalue())
        settings, provenance = load_configuration(ROOT / "configs/experiments/gtzan/late_fusion.yaml", "late_fusion")
        fma, _ = load_configuration(ROOT / "configs/experiments/fma_medium/late_fusion.yaml", "late_fusion")
        for key in ("hidden_dimensions", "dropout", "learning_rate", "batch_size", "epochs", "device", "num_workers"):
            self.assertEqual(plan["settings"][key], settings[key])
            self.assertEqual(settings[key], fma[key])
        self.assertEqual(plan["seeds"], [32, 64, 128, 256, 512])
        self.assertEqual(set(plan["conditions"]), set(provenance["experiment_metadata"]["runs"]))
        self.assertEqual(plan["new_branch_fits_per_seed"], 10)
        self.assertEqual(plan["maximum_new_branch_fits"], 50)
        self.assertEqual(plan["reference_mode"], "read_only_completed_cnn_mixture_1")
        self.assertEqual(plan["summary_directory"], str(self.root / "late"))
        for count in (4, 5, 6):
            self.assertEqual(plan["conditions"][f"mixture_ensemble_{count}"]["trainable_parameters"], count * 735754)
        self.assertFalse((self.root / "late").exists())

    def test_late_validation_requires_completed_m1_without_training(self):
        with patch("stemfusion.experiments._genre.train_branch", side_effect=AssertionError("Must not fit")), \
             contextlib.redirect_stdout(io.StringIO()), \
             self.assertLogs(level="ERROR") as messages:
            status = late_main(self.late_arguments("--seed", "32", "--epochs", "1", "--device", "cpu", "--validate-only"))
        self.assertEqual(status, 2)
        self.assertIn("Completed CNN mixture-only reference required", " ".join(messages.output))
        self.assertFalse((self.root / "late").exists())
        self.assertFalse(self.config.output_directory.exists())

    def test_late_cli_reuses_m1_and_reports_six_ensembles_and_five_stems(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            run_experiment(replace(self.config, selected_run="mixture_only"))
            reference_files = {path: (file_checksum(path), path.stat().st_mtime_ns)
                               for path in self.config.output_directory.rglob("*") if path.is_file()}
            args = self.late_arguments("--seed", "32", "--epochs", "1", "--device", "cpu")
            output = io.StringIO()
            with contextlib.redirect_stdout(output), \
                 patch("stemfusion.experiments._genre.train_branch", wraps=train_branch) as fitting:
                self.assertEqual(late_main(args), 0)
            self.assertEqual(fitting.call_count, 10)
            scheduled = {(call.kwargs["source"], call.kwargs["index"]) for call in fitting.call_args_list}
            self.assertEqual(scheduled, {("mixture", index) for index in range(1, 6)} |
                             {(source, index) for index, source in enumerate(SOURCES) if index})
            root = self.root / "late"
            summary = json.loads((root / "summary_late_fusion_seeds_32.json").read_text())
            self.assertEqual(len(summary["conditions"]), 6)
            self.assertEqual(len(summary["standalone_stems"]), 5)
            self.assertEqual(len(summary["paired_differences"]), 3)
            self.assertEqual(summary["number_of_classes"], 10)
            self.assertEqual(summary["split_counts"], {"train": 10, "validation": 10, "test": 10})
            self.assertIn("MSD_musicnn", output.getvalue())
            self.assertIn("Supporting analysis: individual stem classifiers", output.getvalue())
            self.assertTrue((root / "summary_stem_only_seeds_32.txt").is_file())
            self.assertEqual(len(list(root.rglob("checkpoint.pt"))), 10)
            self.assertFalse((root / "branches/seed_32/mixture_1").exists())
            for index, source in enumerate(SOURCES[1:], start=1):
                metrics = json.loads((root / f"branches/seed_32/{source}/metrics.json").read_text())
                self.assertEqual(metrics["seed"], branch_seed(32, index))
                self.assertEqual(metrics["model"]["input_dimension"], 200)
                self.assertEqual(metrics["model"]["hidden_dimensions"], [1024, 512])
            summed = np.zeros((10, 10), dtype=np.float32)
            for source in SOURCES:
                branch_root = self.config.output_directory if source == "mixture" else root
                branch_name = "mixture_1" if source == "mixture" else source
                with np.load(branch_root / f"branches/seed_32/{branch_name}/predictions.npz", allow_pickle=False) as saved:
                    summed += saved["logits"]
            ensemble_path = Path(summary["runs"]["mixture_all_five_stems"]["32"]["output_directory"])
            with np.load(ensemble_path / "predictions.npz", allow_pickle=False) as saved:
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

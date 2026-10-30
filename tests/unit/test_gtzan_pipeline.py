from __future__ import annotations

import contextlib
import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from stemfusion.cli import build_parser
from stemfusion.configuration import default_config_path, load_configuration
from stemfusion.data.features import FeatureDataError
from stemfusion.data.gtzan import GTZANManifestConfig, LABEL_MAPPING, SOURCES, SPLITS, build_gtzan_manifest, inspect_manifest
from stemfusion.data.gtzan_features import load_gtzan_features
from stemfusion.experiments._extraction import musicfm_config_from_args
from stemfusion.experiments._genre import GenreExperimentConfig, run_experiment, save_stem_results, format_results
from stemfusion.features._musicfm import MusicFMExtractionConfig, extract_musicfm_sources
from stemfusion.training.reporting import file_checksum
from stemfusion.training.trainer import TrainingSettings


class GTZANPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifests/gtzan.jsonl"
        self.features = self.root / "features"
        self.features.mkdir()
        split_files = {}
        for index, split in enumerate(SPLITS):
            path = self.root / f"{split}.txt"
            lines = [f"{genre}/{genre}.{index:05d}.wav" for genre in LABEL_MAPPING]
            if split == "train":
                lines.append("jazz/jazz.00054.wav")
            path.write_text("\n".join(lines) + "\n")
            split_files[split] = path
        self.manifest_config = GTZANManifestConfig(split_files, self.root / "audio", self.root / "stems",
                                                  self.manifest, excluded_track_ids=("jazz.00054",))
        build_gtzan_manifest(self.manifest_config)
        self.records, _ = inspect_manifest(self.manifest)
        for source_index, source in enumerate(SOURCES):
            for split in SPLITS:
                values = {}
                for record in self.records:
                    if record.split == split:
                        vector = np.asarray([record.label / 10, source_index / 6, 0.1, -0.1], dtype=np.float32)
                        values[record.track_id] = torch.from_numpy(vector) if source_index % 2 else vector
                self.write_cache(source, split, values)
        self.write_import_audit()

    def tearDown(self):
        self.temp.cleanup()

    def cache(self, source, split):
        return self.features / f"gtzan_musicfm_{'mix' if source == 'mixture' else source}_{split}.pkl"

    def write_cache(self, source, split, values):
        with self.cache(source, split).open("wb") as stream:
            pickle.dump(values, stream)

    def write_import_audit(self):
        audit = {
            "retained_split_counts": {split: 10 for split in SPLITS}, "label_mapping": LABEL_MAPPING,
            "source_mapping": {source: "mix" if source == "mixture" else source for source in SOURCES},
            "excluded_track_ids": {"train": ["jazz.00054"], "validation": [], "test": []},
            "extraction_source": {"declared_settings": {
                "checkpoint": "pretrained_msd.pt", "statistics": "msd_stats.json", "layer_index": 7,
                "sample_rate": 24000, "channels": "mono", "max_seconds": None,
                "pooling": "temporal_mean", "dimension": 4, "energy_features": False}},
            "caches": {f"{'mix' if source == 'mixture' else source}/{split}": {"sha256": file_checksum(self.cache(source, split))}
                       for source in SOURCES for split in SPLITS},
        }
        (self.features / "import_audit.json").write_text(json.dumps(audit))

    def test_manifest_preserves_splits_labels_and_explicit_exclusion(self):
        self.assertEqual(len(self.records), 30)
        self.assertNotIn("jazz.00054", [record.track_id for record in self.records])
        self.assertEqual(self.records[0].track_id, "blues.00000")
        report = json.loads(self.manifest.with_suffix(".report.json").read_text())
        self.assertFalse(report["audio_files_revalidated"])
        self.assertEqual(report["input_split_counts"]["train"], 11)
        self.assertEqual(report["split_counts"], {split: 10 for split in SPLITS})
        with self.assertRaisesRegex(ValueError, "already exist"):
            build_gtzan_manifest(self.manifest_config)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.manifest_config.split_files["test"].write_text("blues/blues.00000.wav\n")
            build_gtzan_manifest(replace(self.manifest_config, dry_run=True))

    def test_numpy_and_torch_caches_share_format_and_retain_zero_vectors(self):
        data = load_gtzan_features(self.manifest, self.features, dimension=4)
        self.assertEqual(data.number_of_classes, 10)
        for split in SPLITS:
            for source in SOURCES:
                self.assertEqual(data.splits[split].features[source].dtype, np.float32)
                self.assertEqual(data.splits[split].features[source].shape, (10, 4))
        path = self.cache("guitar", "train")
        with path.open("rb") as stream:
            values = pickle.load(stream)
        values["blues.00000"] = torch.zeros(4, dtype=torch.float32)
        self.write_cache("guitar", "train", values)
        self.write_import_audit()
        data = load_gtzan_features(self.manifest, self.features, dimension=4)
        self.assertEqual(data.audit["caches"]["guitar/train"]["zero_vectors"], 1)
        self.assertEqual(len(data.splits["train"].track_ids), 10)

    def test_cache_ids_dtype_dimension_and_finiteness_are_strict(self):
        path = self.cache("piano", "train")
        with path.open("rb") as stream:
            original = pickle.load(stream)
        for vector in (np.ones(4, dtype=np.float64), np.ones(5, dtype=np.float32), np.full(4, np.nan, dtype=np.float32)):
            with self.subTest(vector=vector):
                self.write_cache("piano", "train", {**original, "blues.00000": vector})
                with self.assertRaisesRegex(FeatureDataError, "finite float32"):
                    load_gtzan_features(self.manifest, self.features, 4, require_reports=False)
        missing = dict(original)
        missing.pop("blues.00000")
        self.write_cache("piano", "train", missing)
        with self.assertRaisesRegex(FeatureDataError, "exactly match"):
            load_gtzan_features(self.manifest, self.features, 4, require_reports=False)

    def test_import_provenance_rejects_modified_cache_and_wrong_layer(self):
        with self.cache("mixture", "train").open("ab") as stream:
            stream.write(b"modified")
        with self.assertRaisesRegex(FeatureDataError, "differs from its audit"):
            load_gtzan_features(self.manifest, self.features, 4)
        self.write_import_audit()
        path = self.features / "import_audit.json"
        audit = json.loads(path.read_text())
        audit["extraction_source"]["declared_settings"]["layer_index"] = 12
        path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(FeatureDataError, "MSD/layer-7"):
            load_gtzan_features(self.manifest, self.features, 4)

    def test_shared_profiles_have_identical_training_and_extraction_settings(self):
        parser = build_parser()
        for family in ("mid_fusion", "late_fusion"):
            fma, fma_provenance = load_configuration(default_config_path(f"experiments/fma_medium/{family}.yaml"), family)
            gtzan, gtzan_provenance = load_configuration(default_config_path(f"experiments/gtzan/{family}.yaml"), family)
            ignored = {"manifest", "feature_dir", "output_dir", "summary_dir"}
            self.assertEqual({k: v for k, v in fma.items() if k not in ignored},
                             {k: v for k, v in gtzan.items() if k not in ignored})
            self.assertEqual(fma_provenance["experiment_metadata"]["runs"], gtzan_provenance["experiment_metadata"]["runs"])
        configs = [musicfm_config_from_args(parser.parse_args(["extract", dataset]), source="mixture")
                   for dataset in ("fma-medium", "gtzan")]
        for name in ("sample_rate", "layer_index", "expected_dimension", "temporal_pooling", "is_flash", "max_seconds",
                     "effective_checkpoint_path", "effective_stats_path"):
            self.assertEqual(getattr(configs[0], name), getattr(configs[1], name))
        from stemfusion.experiments._genre import config_from_args
        wrong_dataset = parser.parse_args(["train", "gtzan", "mid-fusion", "--config",
            str(default_config_path("experiments/fma_medium/mid_fusion.yaml"))])
        with self.assertRaisesRegex(ValueError, "YAML dataset"):
            config_from_args(wrong_dataset)

    def test_shared_extractor_writes_gtzan_names_and_training_reads_its_reports(self):
        for record in self.records:
            for value in record.paths.values():
                path = Path(value)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic")
        musicfm = self.root / "musicfm"
        (musicfm / "data").mkdir(parents=True)
        (musicfm / "data/msd_stats.json").write_text("{}")
        (musicfm / "data/pretrained_msd.pt").write_bytes(b"synthetic")
        configs = [MusicFMExtractionConfig(self.manifest, self.features, source, musicfm,
                   dataset="gtzan", device="cpu", expected_dimension=4, overwrite=True) for source in SOURCES]
        with contextlib.redirect_stderr(io.StringIO()):
            summaries = extract_musicfm_sources(configs, feature_extractor=lambda _: torch.ones(4))
        self.assertEqual(sum(summary.succeeded for summary in summaries), 180)
        data = load_gtzan_features(self.manifest, self.features, dimension=4)
        self.assertEqual(len(data.audit["extraction_reports"]), 6)
        self.assertFalse(data.audit["import_provenance"])
        self.assertTrue(self.cache("mixture", "test").is_file())

    def test_all_five_seeds_mid_late_and_standalone_reports_reuse_85_fits(self):
        data = load_gtzan_features(self.manifest, self.features, dimension=4)
        output = self.root / "outputs"
        settings = TrainingSettings(hidden_dimensions=(8, 4), batch_size=10, epochs=1, device="cpu")
        configs = {}
        for family in ("mid_fusion", "late_fusion"):
            _, provenance = load_configuration(default_config_path(f"experiments/gtzan/{family}.yaml"), family)
            configs[family] = GenreExperimentConfig(self.manifest, self.features, output, family,
                provenance["experiment_metadata"]["runs"], (32, 64, 128, 256, 512),
                settings=settings, dimension=4, number_of_classes=10, dataset="gtzan")
        mid = run_experiment(configs["mid_fusion"], data=data)
        late = run_experiment(configs["late_fusion"], data=data)
        self.assertEqual(len(mid["conditions"]), 7)
        self.assertEqual(len(late["conditions"]), 6)
        self.assertEqual(len(late["standalone_stems"]), 5)
        self.assertNotIn("historical_selection_note", late)
        self.assertIn("Dataset: GTZAN | Task: 10-class", format_results(late))
        checkpoints = {path: file_checksum(path) for path in output.rglob("checkpoint.pt")}
        self.assertEqual(len(checkpoints), 85)
        with patch("torch.optim.Adam", side_effect=AssertionError("Completed fits must be reused")):
            for config in configs.values():
                run_experiment(config, data=data)
            with patch("stemfusion.data.gtzan_features.load_gtzan_features", side_effect=AssertionError("Reporting must not load features")):
                standalone = save_stem_results(configs["late_fusion"])
        self.assertEqual(standalone["dataset"], "GTZAN")
        self.assertEqual(checkpoints, {path: file_checksum(path) for path in checkpoints})
        for family in ("mid_fusion", "late_fusion", "stem_only"):
            self.assertTrue((output / f"summary_{family}.txt").is_file())
            self.assertTrue((output / f"summary_{family}.json").is_file())

    def test_direct_script_works_without_pythonpath_and_does_not_train_for_validation(self):
        script = Path(__file__).resolve().parents[2] / "src/stemfusion/experiments/gtzan/train_mid.py"
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        result = subprocess.run([sys.executable, str(script), "--manifest", str(self.manifest),
            "--feature-dir", str(self.features), "--output-dir", str(self.root / "outputs"),
            "--expected-feature-dimension", "4", "--validate-only"],
            cwd=self.root, env=env, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["number_of_classes"], 10)
        self.assertFalse((self.root / "outputs").exists())

    def test_direct_mid_late_and_stem_reporting_commands(self):
        scripts = Path(__file__).resolve().parents[2] / "src/stemfusion/experiments/gtzan"
        output = self.root / "outputs"
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        arguments = ["--manifest", str(self.manifest), "--feature-dir", str(self.features),
            "--output-dir", str(output), "--expected-feature-dimension", "4", "--seed", "32",
            "--hidden-dimensions", "8", "4", "--batch-size", "10", "--epochs", "1", "--device", "cpu"]
        for script in ("train_mid.py", "train_late.py"):
            result = subprocess.run([sys.executable, str(scripts / script), *arguments],
                cwd=self.root, env=env, capture_output=True, text=True, check=True)
            self.assertIn("Dataset: GTZAN | Task: 10-class", result.stdout)
            self.assertNotIn("Study context:", result.stdout)
        checkpoints = {path: file_checksum(path) for path in output.rglob("checkpoint.pt")}
        self.assertEqual(len(checkpoints), 17)
        result = subprocess.run([sys.executable, str(scripts / "train_late.py"), *arguments, "--report-stems-only"],
            cwd=self.root, env=env, capture_output=True, text=True, check=True)
        self.assertIn("Guitar only", result.stdout)
        self.assertEqual(checkpoints, {path: file_checksum(path) for path in checkpoints})


if __name__ == "__main__":
    unittest.main()

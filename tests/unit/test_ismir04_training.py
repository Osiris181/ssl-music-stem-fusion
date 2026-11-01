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
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from stemfusion.cli import build_parser
from stemfusion.configuration import default_config_path, load_configuration
from stemfusion.data.features import FeatureDataError
from stemfusion.data.ismir04 import LABEL_MAPPING, SOURCES
from stemfusion.data.ismir04_features import SPLITS, load_ismir04_features
from stemfusion.experiments._genre import GenreExperimentConfig, config_from_args, format_results, run_experiment, save_stem_results
from stemfusion.features._musicfm import MusicFMExtractionConfig, extract_musicfm_sources
from stemfusion.training.reporting import file_checksum
from stemfusion.training.trainer import TrainingSettings


class ISMIR04TrainingTests(unittest.TestCase):
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
        self.manifest = self.root / "ismir04.jsonl"
        self.features = self.root / "features"
        self.records = []
        self.vectors = {}
        for split in SPLITS:
            for genre, label in LABEL_MAPPING.items():
                track_id = f"{split}__{genre}__synthetic"
                paths = {}
                for index, source in enumerate(SOURCES):
                    path = self.root / "audio" / track_id / f"{source}.mp3"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"synthetic audio; encoder is injected")
                    paths[source] = str(path)
                    vector = np.asarray([label / 6, index / 6, 0.1, -0.1], dtype=np.float32)
                    self.vectors[path] = torch.from_numpy(vector) if index % 2 else vector
                self.records.append({"track_id": track_id, "genre": genre, "label": label,
                                     "split": split, "paths": paths})
        self.manifest.write_text("".join(json.dumps(record) + "\n" for record in self.records))
        musicfm = self.root / "musicfm"
        (musicfm / "data").mkdir(parents=True)
        (musicfm / "data/msd_stats.json").write_text("{}")
        (musicfm / "data/pretrained_msd.pt").write_bytes(b"synthetic checkpoint")
        configs = [MusicFMExtractionConfig(self.manifest, self.features, source, musicfm,
                   dataset="ismir04", device="cpu", expected_dimension=4, chunk_seconds=30)
                   for source in SOURCES]
        with contextlib.redirect_stderr(io.StringIO()):
            extract_musicfm_sources(configs, feature_extractor=lambda path: self.vectors[path])

    def tearDown(self):
        self.temp.cleanup()

    def cache(self, source, split):
        return self.features / f"ismir04_musicfm_{'mix' if source == 'mixture' else source}_{split}.pkl"

    def write_cache(self, source, split, values):
        with self.cache(source, split).open("wb") as stream:
            pickle.dump(values, stream)

    def test_shared_extractor_reports_and_mixed_vectors_align_without_normalization(self):
        path = self.cache("guitar", "train")
        with path.open("rb") as stream:
            values = pickle.load(stream)
        first = next(iter(values))
        values[first] = torch.zeros(4, dtype=torch.float32)
        self.write_cache("guitar", "train", dict(reversed(list(values.items()))))
        data = load_ismir04_features(self.manifest, self.features, dimension=4)
        self.assertEqual(data.number_of_classes, 6)
        self.assertEqual(data.label_mapping, LABEL_MAPPING)
        self.assertEqual(data.audit["split_counts"], dict.fromkeys(SPLITS, 6))
        self.assertEqual(len(data.audit["extraction_reports"]), 6)
        self.assertEqual(data.audit["extraction_policy"]["chunk_seconds"], 30)
        self.assertEqual(data.audit["caches"]["guitar/train"]["zero_vectors"], 1)
        for split in SPLITS:
            records = [record for record in self.records if record["split"] == split]
            self.assertEqual(data.splits[split].track_ids, tuple(record["track_id"] for record in records))
            np.testing.assert_array_equal(data.splits[split].labels, list(LABEL_MAPPING.values()))
            for source in SOURCES:
                matrix = data.splits[split].features[source]
                self.assertEqual(matrix.shape, (6, 4))
                self.assertEqual(matrix.dtype, np.float32)
                for index, record in enumerate(records):
                    expected = np.zeros(4, dtype=np.float32) if source == "guitar" and record["track_id"] == first else self.vectors[Path(record["paths"][source])]
                    np.testing.assert_array_equal(matrix[index], expected)

    def test_cache_ids_dtype_dimension_and_finiteness_are_strict(self):
        with self.cache("piano", "train").open("rb") as stream:
            original = pickle.load(stream)
        first = next(iter(original))
        for vector in (np.ones(4, dtype=np.float64), np.ones(5, dtype=np.float32),
                       np.ones((1, 4), dtype=np.float32), np.full(4, np.nan, dtype=np.float32),
                       np.full(4, np.inf, dtype=np.float32)):
            with self.subTest(vector=vector):
                self.write_cache("piano", "train", {**original, first: vector})
                with self.assertRaisesRegex(FeatureDataError, "finite float32"):
                    load_ismir04_features(self.manifest, self.features, dimension=4)
        for values in ({key: value for key, value in original.items() if key != first},
                       {**original, "unexpected_id": np.ones(4, dtype=np.float32)}):
            self.write_cache("piano", "train", values)
            with self.assertRaisesRegex(FeatureDataError, "exactly match"):
                load_ismir04_features(self.manifest, self.features, dimension=4)

    def test_reports_reject_legacy_settings_missing_sources_and_incomplete_splits(self):
        path = self.features / "extraction_report_musicfm_mixture.json"
        original = json.loads(path.read_text())
        for key, value in (("layer_index", 12), ("chunk_seconds", None), ("chunk_seconds", 20),
                           ("max_seconds", 30), ("chunk_pooling", "equal_chunk_mean"),
                           ("chunk_partition", None), ("failed", 1), ("source", "guitar"),
                           ("checkpoint_path", "/tmp/pretrained_fma.pt"), ("dry_run", True)):
            with self.subTest(key=key, value=value):
                path.write_text(json.dumps({**original, key: value}))
                with self.assertRaises(FeatureDataError):
                    load_ismir04_features(self.manifest, self.features, dimension=4)
        for field in ("selected", "succeeded"):
            report = json.loads(json.dumps(original))
            report["split_summaries"][0][field] -= 1
            path.write_text(json.dumps(report))
            with self.assertRaises(FeatureDataError):
                load_ismir04_features(self.manifest, self.features, dimension=4)
        path.write_text(json.dumps(original))
        fingerprint = load_ismir04_features(self.manifest, self.features, dimension=4).fingerprint
        path.write_text(json.dumps(original, indent=2))
        self.assertNotEqual(fingerprint, load_ismir04_features(self.manifest, self.features, dimension=4).fingerprint)
        path.unlink()
        with self.assertRaisesRegex(FeatureDataError, "Missing ISMIR04 extraction report"):
            load_ismir04_features(self.manifest, self.features, dimension=4)

    def test_all_three_datasets_resolve_identical_training_settings_and_conditions(self):
        parser = build_parser()
        for family in ("mid_fusion", "late_fusion"):
            reference, reference_provenance = load_configuration(default_config_path(f"experiments/fma_medium/{family}.yaml"), family)
            for dataset, classes in (("gtzan", 10), ("ismir04", 6)):
                values, provenance = load_configuration(default_config_path(f"experiments/{dataset}/{family}.yaml"), family)
                ignored = {"manifest", "feature_dir", "output_dir", "summary_dir"}
                self.assertEqual({key: value for key, value in reference.items() if key not in ignored},
                                 {key: value for key, value in values.items() if key not in ignored})
                self.assertEqual(reference_provenance["experiment_metadata"]["runs"], provenance["experiment_metadata"]["runs"])
                config = config_from_args(parser.parse_args(["train", dataset, family.replace("_", "-")]))
                self.assertEqual(config.number_of_classes, classes)
                self.assertEqual(config.seeds, (32, 64, 128, 256, 512))
                self.assertEqual(config.settings, TrainingSettings())
        wrong = parser.parse_args(["train", "ismir04", "mid-fusion", "--config",
                                  str(default_config_path("experiments/gtzan/mid_fusion.yaml"))])
        with self.assertRaisesRegex(ValueError, "YAML dataset"):
            config_from_args(wrong)

    def test_five_seed_mid_late_and_stem_reports_share_85_fits_and_reuse_checkpoints(self):
        data = load_ismir04_features(self.manifest, self.features, dimension=4)
        output = self.root / "outputs"
        summaries = self.root / "summaries"
        settings = TrainingSettings(hidden_dimensions=(8, 4), batch_size=6, epochs=1, device="cpu")
        configs = {}
        for family in ("mid_fusion", "late_fusion"):
            _, provenance = load_configuration(default_config_path(f"experiments/ismir04/{family}.yaml"), family)
            configs[family] = GenreExperimentConfig(self.manifest, self.features, output, family,
                provenance["experiment_metadata"]["runs"], (32, 64, 128, 256, 512),
                settings=settings, dimension=4, number_of_classes=6, dataset="ismir04", summary_directory=summaries)
        mid = run_experiment(configs["mid_fusion"], data=data)
        late = run_experiment(configs["late_fusion"], data=data)
        self.assertEqual(len(mid["conditions"]), 7)
        self.assertEqual(len(late["conditions"]), 6)
        self.assertEqual(len(late["standalone_stems"]), 5)
        self.assertEqual(mid["conditions"]["mixture_only"], late["mixture_only_reference"])
        self.assertNotIn("historical_selection_note", late)
        self.assertIn("Dataset: ISMIR04 | Task: 6-class", format_results(late))
        checkpoints = {path: file_checksum(path) for path in output.rglob("checkpoint.pt")}
        self.assertEqual(len(checkpoints), 85)
        with patch("torch.optim.Adam", side_effect=AssertionError("Completed fits must be reused")):
            for config in configs.values():
                run_experiment(config, data=data)
            with patch("stemfusion.data.ismir04_features.load_ismir04_features", side_effect=AssertionError("Reporting must not load features")):
                standalone = save_stem_results(configs["late_fusion"])
        self.assertEqual(standalone["dataset"], "ISMIR04")
        self.assertEqual(checkpoints, {path: file_checksum(path) for path in checkpoints})
        for family in ("mid_fusion", "late_fusion", "stem_only"):
            self.assertTrue((summaries / f"summary_{family}.json").is_file())
            text = (summaries / f"summary_{family}.txt").read_text()
            self.assertIn("Dataset: ISMIR04 | Task: 6-class", text)
            self.assertNotIn("Study context:", text)
        self.assertIn("Guitar only", (summaries / "summary_late_fusion.txt").read_text())

    def test_direct_mid_late_validation_training_and_saved_stem_reporting(self):
        scripts = Path(__file__).resolve().parents[2] / "src/stemfusion/experiments/ismir04"
        output = self.root / "outputs"
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        arguments = ["--manifest", str(self.manifest), "--feature-dir", str(self.features),
                     "--output-dir", str(output), "--expected-feature-dimension", "4", "--device", "cpu"]
        for script in ("train_mid.py", "train_late.py"):
            result = subprocess.run([sys.executable, str(scripts / script), *arguments, "--validate-only"],
                cwd=self.root, env=env, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)["number_of_classes"], 6)
            self.assertFalse(output.exists())
        arguments.extend(["--seed", "32", "--hidden-dimensions", "8", "4", "--batch-size", "6", "--epochs", "1"])
        for script in ("train_mid.py", "train_late.py"):
            result = subprocess.run([sys.executable, str(scripts / script), *arguments],
                cwd=self.root, env=env, capture_output=True, text=True, check=True)
            self.assertIn("Dataset: ISMIR04 | Task: 6-class", result.stdout)
        checkpoints = {path: file_checksum(path) for path in output.rglob("checkpoint.pt")}
        self.assertEqual(len(checkpoints), 17)
        result = subprocess.run([sys.executable, str(scripts / "train_late.py"), *arguments, "--report-stems-only"],
            cwd=self.root, env=env, capture_output=True, text=True, check=True)
        self.assertIn("Guitar only", result.stdout)
        self.assertEqual(checkpoints, {path: file_checksum(path) for path in checkpoints})


if __name__ == "__main__":
    unittest.main()

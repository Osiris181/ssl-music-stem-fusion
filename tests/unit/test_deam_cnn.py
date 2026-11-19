"""DEAM preserves native scores through the shared PMEmo CNN extraction flow."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from stemfusion.experiments import _cnn_extraction as cnn
from stemfusion.experiments.deam import extraction_cnn as entry

ROOT = Path(__file__).resolve().parents[2]


class DEAMCNNTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "deam.jsonl"
        self.rows = [
            {"dataset": "deam", "track_id": str(index), "relative_path": f"{index}.mp3",
             "split": split, "annotations": {"valence": 5.75, "arousal": 4.0},
             "paths": {source: str(self.root / str(index) / f"{source}.mp3") for source in cnn.SOURCES}}
            for index, split in enumerate(("train", "validation", "test"), 1)
        ]
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")

    def test_original_reader_preserves_native_scores_and_source_mapping(self):
        self.assertEqual(cnn.read_records(self.manifest, dataset="deam"), self.rows)
        self.assertEqual(self.rows[0]["annotations"], {"valence": 5.75, "arousal": 4.0})
        self.assertNotIn("label", self.rows[0])
        self.assertIs(entry.cache_valid, cnn.cache_valid)
        self.assertIs(entry.read_records, cnn.read_records)
        for row in self.rows:
            row["annotations"] = {"valence": 1, "arousal": 9}
        self.write_manifest()
        self.assertEqual(cnn.read_records(self.manifest, dataset="deam"), self.rows)

    def test_invalid_native_scores_ids_paths_and_sources_are_rejected(self):
        original = json.loads(json.dumps(self.rows))
        changes = [
            {"annotations": {"valence": .575, "arousal": .4}},
            {"annotations": {"valence": 10, "arousal": 4}},
            {"annotations": {"valence": 5}},
            {"annotations": {"valence": True, "arousal": 4}},
            {"dataset": "pmemo"}, {"track_id": "0001"},
            {"relative_path": "../1.mp3"},
            {"paths": {"mixture": self.rows[0]["paths"]["mixture"]}},
            {"paths": {source: f"relative/{source}.mp3" for source in cnn.SOURCES}},
        ]
        for changed in changes:
            with self.subTest(changed=changed):
                self.rows = json.loads(json.dumps(original))
                self.rows[0].update(changed)
                self.write_manifest()
                with self.assertRaises(ValueError):
                    cnn.read_records(self.manifest, dataset="deam")
        self.rows = [original[0], original[0], *original[1:]]
        self.write_manifest()
        with self.assertRaises(ValueError):
            cnn.read_records(self.manifest, dataset="deam")

    def test_existing_three_splits_are_required(self):
        self.rows = self.rows[:2]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "retain all three existing splits"):
            cnn.read_records(self.manifest, dataset="deam")

    def test_runtime_alignment_and_paths_match_pmemo(self):
        settings = yaml.safe_load(entry.DEFAULT_CONFIG.read_text())
        reference = yaml.safe_load((ROOT / "configs/experiments/pmemo/extraction_cnn.yaml").read_text())
        self.assertEqual(settings["runtime"], reference["runtime"])
        self.assertEqual(settings["audio"], reference["audio"])
        self.assertEqual(settings["dataset"], "deam")
        paths = {key: (entry.DEFAULT_CONFIG.parent / value).resolve() for key, value in settings["paths"].items()}
        self.assertEqual(paths["manifest"], ROOT / "data/manifests/deam/deam_features.jsonl")
        self.assertEqual(paths["output_dir"], ROOT / "data/features/deam/msd_musicnn")
        self.assertEqual(paths["report"], ROOT / "outputs/deam/cnn/extraction_report.json")

    def test_direct_dry_run_without_pythonpath_does_not_touch_audio_or_outputs(self):
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run([
            sys.executable, str(Path(entry.__file__)), "--manifest", str(self.manifest),
            "--output-dir", str(self.root / "features"), "--report", str(self.root / "report.json"),
            "--split", "test", "--dry-run",
        ], cwd=self.root, env=environment, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("DEAM CNN: {'test': 1}; 6 views", completed.stdout)
        self.assertFalse((self.root / "features").exists())
        self.assertFalse((self.root / "report.json").exists())


if __name__ == "__main__":
    unittest.main()

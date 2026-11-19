"""PMEmo uses the shared frozen CNN flow with its original continuous targets."""
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
from stemfusion.experiments.pmemo import extraction_cnn as entry

ROOT = Path(__file__).resolve().parents[2]


class PMEmoCNNTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "pmemo.jsonl"
        self.rows = [
            {"dataset": "pmemo", "track_id": str(index), "relative_path": f"{index}.mp3",
             "split": split, "annotations": {"valence": .575, "arousal": .4},
             "duration_seconds": 30.1,
             "paths": {source: str(self.root / str(index) / f"{source}.mp3") for source in cnn.SOURCES}}
            for index, split in enumerate(("train", "validation", "test"), 1)
        ]
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")

    def test_existing_identity_and_scores_are_preserved_by_the_shared_reader(self):
        records = cnn.read_records(self.manifest, dataset="pmemo")
        self.assertEqual(records, self.rows)
        self.assertEqual(records[0]["track_id"], "1")
        self.assertEqual(records[0]["annotations"], {"valence": .575, "arousal": .4})
        self.assertNotIn("label", records[0])
        self.assertIs(entry.cache_valid, cnn.cache_valid)
        self.assertIs(entry.read_records, cnn.read_records)

    def test_invalid_scores_ids_and_missing_sources_are_rejected(self):
        original = json.loads(json.dumps(self.rows))
        changes = [
            {"annotations": {"valence": 5.75, "arousal": .4}},
            {"annotations": {"valence": .5}},
            {"dataset": "mtg_jamendo"},
            {"track_id": "0001"},
            {"relative_path": "../1.mp3"},
            {"paths": {"mixture": self.rows[0]["paths"]["mixture"]}},
        ]
        for changed in changes:
            with self.subTest(changed=changed):
                self.rows = json.loads(json.dumps(original))
                self.rows[0].update(changed)
                self.write_manifest()
                with self.assertRaises(ValueError):
                    cnn.read_records(self.manifest, dataset="pmemo")
        self.rows = [original[0], original[0], *original[1:]]
        self.write_manifest()
        with self.assertRaises(ValueError):
            cnn.read_records(self.manifest, dataset="pmemo")

    def test_missing_splits_are_not_silently_accepted(self):
        self.rows = self.rows[:2]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "retain all three existing splits"):
            cnn.read_records(self.manifest, dataset="pmemo")

    def test_runtime_and_alignment_settings_match_the_existing_cnn_datasets(self):
        settings = yaml.safe_load(entry.DEFAULT_CONFIG.read_text())
        for dataset in ("gtzan", "ismir04", "fma_medium", "mtg_jamendo"):
            reference = yaml.safe_load((ROOT / f"configs/experiments/{dataset}/extraction_cnn.yaml").read_text())
            self.assertEqual(settings["runtime"], reference["runtime"])
            self.assertEqual(settings["audio"], reference["audio"])
        self.assertEqual(settings["dataset"], "pmemo")
        paths = {key: (entry.DEFAULT_CONFIG.parent / value).resolve() for key, value in settings["paths"].items()}
        self.assertEqual(paths["manifest"], ROOT / "data/manifests/pmemo/pmemo_features.jsonl")
        self.assertEqual(paths["output_dir"], ROOT / "data/features/pmemo/msd_musicnn")
        self.assertEqual(paths["report"], ROOT / "outputs/pmemo/cnn/extraction_report.json")

    def test_direct_dry_run_without_pythonpath_never_reads_audio_or_creates_outputs(self):
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run([
            sys.executable, str(Path(entry.__file__)), "--manifest", str(self.manifest),
            "--output-dir", str(self.root / "features"), "--report", str(self.root / "report.json"),
            "--split", "test", "--dry-run",
        ], cwd=self.root, env=environment, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PMEMO CNN: {'test': 1}; 6 views", completed.stdout)
        self.assertFalse((self.root / "features").exists())
        self.assertFalse((self.root / "report.json").exists())


if __name__ == "__main__":
    unittest.main()

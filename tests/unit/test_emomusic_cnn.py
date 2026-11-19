"""EmoMusic uses the shared emotion CNN extractor without changing labels or audio mapping."""
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
from stemfusion.experiments.emomusic import extraction_cnn as entry

ROOT = Path(__file__).resolve().parents[2]


class EmoMusicCNNTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "emomusic.jsonl"
        self.rows = [
            {"dataset": "emomusic", "track_id": str(index), "relative_path": f"{index}.mp3",
             "split": split, "annotations": {"valence": 4.3, "arousal": 7.2},
             "paths": {"mixture": str(self.root / "emoMusic/clips_45sec" / f"{index}.mp3"),
                       **{source: str(self.root / "DEAM_separated/htdemucs_6s" / str(index) / f"{source}.mp3")
                          for source in cnn.SOURCES if source != "mixture"}}}
            for index, split in enumerate(("train", "validation", "test"), 1)
        ]
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")

    def test_supplied_columns_ids_and_shared_deam_stem_mapping_are_preserved(self):
        before = self.manifest.read_bytes()
        records = cnn.read_records(self.manifest, dataset="emomusic")
        self.assertEqual(records, self.rows)
        self.assertEqual(records[0]["annotations"], {"valence": 4.3, "arousal": 7.2})
        self.assertIn("emoMusic/clips_45sec", records[0]["paths"]["mixture"])
        self.assertIn("DEAM_separated/htdemucs_6s", records[0]["paths"]["guitar"])
        self.assertNotIn("label", records[0])
        self.assertEqual(before, self.manifest.read_bytes())
        self.assertIs(entry.cache_valid, cnn.cache_valid)
        self.assertIs(entry.read_records, cnn.read_records)
        for row in self.rows:
            row["annotations"] = {"valence": 1, "arousal": 9}
        self.write_manifest()
        self.assertEqual(cnn.read_records(self.manifest, dataset="emomusic"), self.rows)

    def test_invalid_scores_ids_paths_and_missing_sources_are_rejected(self):
        original = json.loads(json.dumps(self.rows))
        changes = [
            {"annotations": {"valence": .43, "arousal": .72}},
            {"annotations": {"valence": 10, "arousal": 7.2}},
            {"annotations": {"valence": 4.3}},
            {"annotations": {"valence": True, "arousal": 7.2}},
            {"dataset": "deam"}, {"track_id": "0001"}, {"relative_path": "../1.mp3"},
            {"paths": {"mixture": self.rows[0]["paths"]["mixture"]}},
            {"paths": {source: f"relative/{source}.mp3" for source in cnn.SOURCES}},
        ]
        for changed in changes:
            with self.subTest(changed=changed):
                self.rows = json.loads(json.dumps(original))
                self.rows[0].update(changed)
                self.write_manifest()
                with self.assertRaises(ValueError):
                    cnn.read_records(self.manifest, dataset="emomusic")
        self.rows = [original[0], original[0], *original[1:]]
        self.write_manifest()
        with self.assertRaises(ValueError):
            cnn.read_records(self.manifest, dataset="emomusic")

    def test_all_existing_splits_are_required(self):
        self.rows = self.rows[:2]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "retain all three existing splits"):
            cnn.read_records(self.manifest, dataset="emomusic")

    def test_settings_match_pmemo_and_deam_with_separate_emomusic_outputs(self):
        settings = yaml.safe_load(entry.DEFAULT_CONFIG.read_text())
        for dataset in ("pmemo", "deam"):
            reference = yaml.safe_load((ROOT / f"configs/experiments/{dataset}/extraction_cnn.yaml").read_text())
            self.assertEqual(settings["runtime"], reference["runtime"])
            self.assertEqual(settings["audio"], reference["audio"])
        self.assertEqual(settings["dataset"], "emomusic")
        paths = {key: (entry.DEFAULT_CONFIG.parent / value).resolve() for key, value in settings["paths"].items()}
        self.assertEqual(paths["manifest"], ROOT / "data/manifests/emomusic/emomusic_features.jsonl")
        self.assertEqual(paths["output_dir"], ROOT / "data/features/emomusic/msd_musicnn")
        self.assertEqual(paths["report"], ROOT / "outputs/emomusic/cnn/extraction_report.json")

    def test_direct_dry_run_without_pythonpath_does_not_decode_or_write(self):
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run([
            sys.executable, str(Path(entry.__file__)), "--manifest", str(self.manifest),
            "--output-dir", str(self.root / "features"), "--report", str(self.root / "report.json"),
            "--split", "test", "--dry-run",
        ], cwd=self.root, env=environment, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("EMOMUSIC CNN: {'test': 1}; 6 views", completed.stdout)
        self.assertFalse((self.root / "features").exists())
        self.assertFalse((self.root / "report.json").exists())


if __name__ == "__main__":
    unittest.main()

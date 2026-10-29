from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from stemfusion.configuration import default_config_path


class DirectExtractionScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.script = (
            Path(__file__).resolve().parents[2]
            / "src/stemfusion/experiments/fma_medium/extraction.py"
        )
        self.environment = dict(os.environ)
        self.environment.pop("PYTHONPATH", None)
        self.environment["PYTHONDONTWRITEBYTECODE"] = "1"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def run_script(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(self.script), *arguments],
            cwd=self.root, env=self.environment,
            capture_output=True, text=True, timeout=30,
        )

    def test_help_works_outside_checkout_without_pythonpath(self) -> None:
        result = self.run_script("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: extraction.py", result.stdout)
        self.assertIn("--config", result.stdout)

    def test_direct_dry_run_loads_yaml_sources_without_model_or_output(self) -> None:
        model_data = self.root / "musicfm/data"
        model_data.mkdir(parents=True)
        (model_data / "msd_stats.json").write_text("{}")
        (model_data / "pretrained_msd.pt").write_bytes(b"synthetic")
        manifest = self.root / "manifest.jsonl"
        manifest.write_text(json.dumps({
            "track_id": "000002", "split": "train", "genre": "Rock",
            "label": 0, "paths": {"mixture": "unused.mp3", "guitar": "unused.mp3"},
        }) + "\n")
        config = self.root / "extraction.yaml"
        output = self.root / "output"
        config.write_text(yaml.safe_dump({
            "feature_profile": str(default_config_path("features/musicfm_msd_layer7.yaml")),
            "sources": ["mixture", "guitar"],
            "paths": {
                "manifest": str(manifest),
                "musicfm_root": str(self.root / "musicfm"),
                "output_dir": str(output),
            },
            "output": {"files_per_source": ["train"]},
        }))
        result = self.run_script("--config", str(config), "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("Source 1/2: mixture", result.stderr)
        self.assertIn("Source 2/2: guitar", result.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

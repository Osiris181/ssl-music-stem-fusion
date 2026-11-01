from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import yaml

from stemfusion.cli import build_parser
from stemfusion.configuration import default_config_path, load_configuration
from stemfusion.separation.htdemucs import (
    HTDEMUCS_6S_STEMS, HTDemucsConfigurationError, HTDemucsSeparationConfig,
    separate_manifest,
)


class DatasetSeparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "outputs"
        self.manifest = self.root / "manifest.jsonl"
        self.originals = []
        self.records = []
        for index in range(2):
            source = self.root / "audio" / f"artist_{index}" / "same_name.mp3"
            source.parent.mkdir(parents=True)
            source.write_bytes(f"original recording {index}".encode())
            self.originals.append(source)
            destination = self.output / ("legacy" if index == 0 else "repairs") / "htdemucs_6s" / f"unique_{index}"
            self.records.append({"track_id": f"record_{index}", "split": "test", "genre": "classical", "label": 0,
                "paths": {"mixture": str(source), **{stem: str(destination / f"{stem}.mp3") for stem in HTDEMUCS_6S_STEMS}}})
        self.write_manifest()
        self.config = HTDemucsSeparationConfig((), self.output)

    def tearDown(self):
        self.temporary.cleanup()

    def write_manifest(self):
        self.manifest.write_text("".join(json.dumps(record) + "\n" for record in self.records))

    def complete(self, index):
        for stem in HTDEMUCS_6S_STEMS:
            path = Path(self.records[index]["paths"][stem])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"stem")

    def test_manifest_stages_unique_names_and_reuses_exact_destinations(self):
        self.complete(0)
        before = [path.read_bytes() for path in self.originals]
        staged = []

        def runner(command, **kwargs):
            alias = Path(command[-1])
            staged.append(alias)
            self.assertTrue(alias.is_symlink())
            self.assertEqual(alias.stem, "unique_1")
            self.assertEqual(alias.resolve(), self.originals[1])
            out = Path(command[command.index("--out") + 1])
            for stem in HTDEMUCS_6S_STEMS:
                path = out / "htdemucs_6s" / alias.stem / f"{stem}.mp3"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"new stem")
            return subprocess.CompletedProcess(command, 0)

        summary = separate_manifest(self.config, self.manifest, runner=runner)
        self.assertEqual((summary.discovered, summary.skipped_existing, summary.succeeded), (2, 1, 1))
        self.assertEqual(summary.planned, 1)
        self.assertEqual(len(summary.manifest_jobs), 1)
        self.assertFalse(any(path.parent.exists() for path in staged))
        self.assertEqual(before, [path.read_bytes() for path in self.originals])
        with patch("stemfusion.separation.htdemucs.tempfile.TemporaryDirectory", side_effect=AssertionError("Completed jobs must not stage audio")):
            reused = separate_manifest(self.config, self.manifest, runner=runner)
        self.assertEqual(reused.skipped_existing, 2)
        self.assertEqual(reused.planned, 0)

    def test_manifest_dry_run_has_no_writes_or_staging(self):
        with patch("stemfusion.separation.htdemucs.tempfile.TemporaryDirectory", side_effect=AssertionError("Dry runs must not stage audio")):
            summary = separate_manifest(replace(self.config, dry_run=True), self.manifest)
        self.assertEqual(summary.planned, 2)
        self.assertFalse(self.output.exists())

    def test_manifest_rejects_destination_collisions_and_unexpected_roots(self):
        self.records[1]["paths"].update({stem: self.records[0]["paths"][stem] for stem in HTDEMUCS_6S_STEMS})
        self.write_manifest()
        with self.assertRaisesRegex(HTDemucsConfigurationError, "share a stem destination"):
            separate_manifest(replace(self.config, dry_run=True), self.manifest)
        with self.assertRaisesRegex(HTDemucsConfigurationError, "share a stem destination"):
            separate_manifest(replace(self.config, dry_run=True, limit=1), self.manifest)
        self.records = self.records[:1]
        self.records[0]["paths"].update({stem: str(self.root / "outside/htdemucs_6s/track" / f"{stem}.mp3") for stem in HTDEMUCS_6S_STEMS})
        self.write_manifest()
        with self.assertRaisesRegex(HTDemucsConfigurationError, "under --output-root"):
            separate_manifest(replace(self.config, dry_run=True), self.manifest)

    def test_failed_job_reports_original_path_and_cleans_staging(self):
        self.records = self.records[:1]
        self.write_manifest()
        staged = []

        def runner(command, **kwargs):
            staged.append(Path(command[-1]))
            return subprocess.CompletedProcess(command, 1)

        summary = separate_manifest(self.config, self.manifest, runner=runner)
        self.assertEqual(summary.failed, 1)
        self.assertEqual(summary.failures[0].input_path, str(self.originals[0]))
        self.assertFalse(staged[0].parent.exists())

    def test_dataset_presets_inherit_identical_model_settings(self):
        settings = {}
        for dataset in ("fma_medium", "gtzan", "ismir04"):
            defaults, provenance = load_configuration(default_config_path(f"experiments/{dataset}/separation.yaml"), "separation")
            settings[dataset] = {key: value for key, value in defaults.items()
                if key not in ("input_root", "output_root", "report", "manifest", "extensions", "batch_size")}
            self.assertEqual(provenance["experiment_metadata"]["dataset"], dataset)
        self.assertEqual(settings["fma_medium"], settings["gtzan"])
        self.assertEqual(settings["fma_medium"], settings["ismir04"])
        args = build_parser().parse_args(["separate", "htdemucs", "--config",
            str(default_config_path("experiments/ismir04/separation.yaml")), "--dry-run"])
        self.assertIsNotNone(args.manifest)
        self.assertIsNone(args.input_root)

    def test_each_direct_script_runs_without_pythonpath(self):
        repo = Path(__file__).resolve().parents[2]
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        input_root = self.root / "simple_input"
        input_root.mkdir()
        (input_root / "track.mp3").write_bytes(b"synthetic audio")
        for dataset in ("fma_medium", "gtzan", "ismir04"):
            with self.subTest(dataset=dataset):
                custom = self.root / f"{dataset}.yaml"
                custom.write_text(yaml.safe_dump({"dataset": dataset,
                    "separator_profile": str(default_config_path("separators/htdemucs_6s.yaml")),
                    "paths": {"input_root": str(input_root), "output_root": str(self.output)}}))
                result = subprocess.run([sys.executable, str(repo / f"src/stemfusion/experiments/{dataset}/separation.py"),
                    "--config", str(custom), "--dry-run"], cwd=self.root, env=env, text=True, capture_output=True, check=True)
                data = json.loads(result.stdout)
                self.assertEqual(data["planned"], 1)
                self.assertEqual(data["resolved_config"]["model_name"], "htdemucs_6s")
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()

"""FMA's direct genre-profile inheritance preserves settings and trained runs."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from stemfusion.cli import build_parser
from stemfusion.configuration import default_config_path, load_configuration
from stemfusion.data.fma_features import SOURCES, load_fma_features
from stemfusion.experiments._genre import config_from_args, run_experiment
from stemfusion.training.reporting import file_checksum


class FMAConfigurationLayoutTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repository = default_config_path("training/genre.yaml").parents[2]
        self.legacy_directory = self.root / "legacy"
        self.legacy_directory.mkdir()
        self.write_yaml("fma_medium_genre.yaml", {
            "training_profile": str(default_config_path("training/genre.yaml")),
            "task": {"number_of_classes": 16},
            "paths": {
                "manifest": str(self.repository / "data/manifests/fma_medium/fma_medium_htdemucs.jsonl"),
                "feature_dir": str(self.repository / "data/features/fma_medium/musicfm_msd_layer7"),
                "output_dir": str(self.repository / "outputs/fma_medium/phase2"),
                "summary_dir": str(self.repository / "outputs/fma_medium"),
            },
        })
        for family, fusion, model in (
            ("mid_fusion", "concatenate", "deep_mlp"),
            ("late_fusion", "independent_logits_then_equal_sum", "per_source_deep_mlp"),
        ):
            self.write_yaml(f"fma_medium_{family}.yaml", {
                "training_profile": "fma_medium_genre.yaml",
                "features": {"fusion": fusion}, "model": {"name": model},
            })
            experiment = yaml.safe_load(self.experiment(family).read_text())
            for name in ("paths", "features", "model"):
                experiment.pop(name)
            experiment.update({
                "training_profile": f"fma_medium_{family}.yaml",
                "feature_profile": str(default_config_path("features/musicfm_msd_layer7.yaml")),
                "task": "genre_classification",
            })
            self.write_yaml(f"{family}.yaml", experiment)

    def write_yaml(self, filename, values):
        path = self.legacy_directory / filename
        path.write_text(yaml.safe_dump(values), encoding="utf-8")
        return path

    def experiment(self, family):
        return default_config_path(f"experiments/fma_medium/{family}.yaml")

    def test_resolved_settings_and_named_runs_equal_the_legacy_hierarchy(self):
        for family in ("mid_fusion", "late_fusion"):
            current = self.experiment(family)
            legacy = self.legacy_directory / f"{family}.yaml"
            _, provenance = load_configuration(current, family)
            names = [None, *provenance["experiment_metadata"]["runs"]]
            for run in names:
                with self.subTest(family=family, run=run):
                    old_defaults, old_provenance = load_configuration(legacy, family, run)
                    new_defaults, new_provenance = load_configuration(current, family, run)
                    self.assertEqual(old_defaults, new_defaults)
                    for key in ("yaml_settings_before_cli_overrides", "experiment_metadata", "run_name"):
                        self.assertEqual(old_provenance[key], new_provenance[key])
                    files = {Path(item["path"]) for item in new_provenance["config_files"]}
                    self.assertEqual(files, {
                        current, default_config_path("training/genre.yaml"),
                        default_config_path("features/musicfm_msd_layer7.yaml"),
                    })
            for name in ("genre", "mid_fusion", "late_fusion"):
                self.assertFalse((self.repository / f"configs/training/fma_medium_{name}.yaml").exists())

    def synthetic_features(self):
        manifest = self.root / "manifest.jsonl"
        features = self.root / "features"
        features.mkdir()
        rows = [
            {"track_id": f"{split_index * 16 + label + 1:06d}", "split": split,
             "genre": f"genre_{label}", "label": label, "paths": {"mixture": "/unused.mp3"}}
            for split_index, split in enumerate(("train", "validation", "test"))
            for label in range(16)
        ]
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        for index, source in enumerate(SOURCES):
            for split in ("train", "validation", "test"):
                vectors = {
                    row["track_id"]: np.asarray([row["label"] / 16, index / 6, 0.1, -0.1], np.float32)
                    for row in rows if row["split"] == split
                }
                with (features / f"fma_musicfm_{source}_{split}.pkl").open("wb") as handle:
                    pickle.dump(vectors, handle)
        return manifest, features, load_fma_features(manifest, features, 4, 16, require_reports=False)

    def test_legacy_mid_late_and_stem_runs_resume_without_rewriting_artifacts(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        manifest, features, data = self.synthetic_features()
        output = self.root / "runs"
        arguments = [
            "--manifest", str(manifest), "--feature-dir", str(features),
            "--output-dir", str(output), "--expected-feature-dimension", "4",
            "--hidden-dimensions", "8", "4", "--batch-size", "16", "--epochs", "1",
            "--seed", "32", "--device", "cpu", "--num-workers", "0",
        ]
        configurations = {}
        results = {}
        for family in ("mid_fusion", "late_fusion"):
            command = ["train", "fma-medium", family.replace("_", "-")]
            legacy = config_from_args(build_parser().parse_args([
                *command, "--config", str(self.legacy_directory / f"{family}.yaml"), *arguments,
            ]))
            current = config_from_args(build_parser().parse_args([*command, *arguments]))
            old_values, new_values = asdict(legacy), asdict(current)
            old_values.pop("provenance")
            new_values.pop("provenance")
            self.assertEqual(old_values, new_values)
            self.assertNotEqual(legacy.provenance["config_files"], current.provenance["config_files"])
            results[family] = run_experiment(legacy, data=data)
            configurations[family] = current
        self.assertEqual(len(results["mid_fusion"]["conditions"]), 7)
        self.assertEqual(len(results["late_fusion"]["conditions"]), 6)
        self.assertEqual(len(results["late_fusion"]["standalone_stems"]), 5)
        directories = [path.parent for path in output.rglob("run_manifest.json")]
        self.assertEqual(len(directories), 23)  # 17 selected models and six ensembles.
        before = {path: (file_checksum(path), path.stat().st_mtime_ns)
                  for directory in directories for path in directory.iterdir() if path.is_file()}
        with patch("torch.optim.Adam", side_effect=AssertionError("Compatible runs must not retrain")):
            for family, config in configurations.items():
                self.assertEqual(run_experiment(config, data=data), results[family])
        self.assertEqual(before, {path: (file_checksum(path), path.stat().st_mtime_ns) for path in before})
        config = configurations["mid_fusion"]
        changed = replace(config, settings=replace(config.settings, epochs=2))
        with self.assertRaisesRegex(ValueError, "incompatible settings"):
            run_experiment(changed, data=data)
        self.assertEqual(before, {path: (file_checksum(path), path.stat().st_mtime_ns) for path in before})


if __name__ == "__main__":
    unittest.main()

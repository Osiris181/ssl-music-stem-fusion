"""Shared training profiles must preserve the completed Jamendo protocol."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from stemfusion.cli import build_parser
from stemfusion.configuration import ConfigurationError, default_config_path, load_configuration


class Music2EmoConfigurationTests(unittest.TestCase):
    def test_flattened_reproduction_resolves_shared_task_and_common_profiles(self):
        preset = default_config_path("experiments/mtg_jamendo/reproduction.yaml")
        values, provenance = load_configuration(preset, "jamendo_reproduction")
        self.assertEqual([Path(item["path"]).name for item in provenance["config_files"]],
                         ["reproduction.yaml", "mood_theme_tagging.yaml", "music2emo.yaml"])
        expected = {"epochs": 200, "batch_size": 8, "learning_rate": 1e-4,
                    "early_stopping_patience": 10, "early_stopping_min_delta": 1e-4,
                    "seeds": [32, 64, 128, 256, 512], "device": "cuda", "num_workers": 4,
                    "resume": True, "overwrite": False, "validate_only": False, "dry_run": False}
        for name, value in expected.items():
            self.assertEqual(values[name], value, name)
        snapshot = provenance["yaml_settings_before_cli_overrides"]
        self.assertEqual(snapshot["features"]["source"], "mixture")
        self.assertEqual(snapshot["model"]["hidden_dimensions"], [512, 256])
        self.assertEqual(snapshot["model"]["output_labels"], 56)
        self.assertEqual(snapshot["optimization"]["early_stopping"]["monitor"], "val_loss")
        self.assertEqual(snapshot["checkpoint"]["minimize"], "weighted_bce_loss")
        self.assertEqual(snapshot["checkpoint"]["save_top_k"], 5)
        self.assertTrue(snapshot["checkpoint"]["save_last"])
        self.assertEqual(snapshot["evaluation"], {"output_transform": "sigmoid", "roc_auc": "macro",
                                                   "pr_auc": "macro_average_precision"})

    def test_flattened_configs_keep_existing_data_and_artifact_paths(self):
        configs = default_config_path("experiments/mtg_jamendo/reproduction.yaml").parents[2]
        root = configs.parent
        values, _ = load_configuration(configs / "experiments/mtg_jamendo/reproduction.yaml", "jamendo_reproduction")
        expected = {"manifest": "data/manifests/mtg_jamendo/mtg_jamendo_features.jsonl",
                    "mert_dir": "data/features/mtg_jamendo/mert_chord_key/mert_v1_95m",
                    "harmony_dir": "data/features/mtg_jamendo/harmony",
                    "tag_list": "data/manifests/mtg_jamendo/tag_list.npy",
                    "author_metadata": "data/manifests/mtg_jamendo/music2emotion",
                    "output_dir": "outputs/mtg_jamendo/mert_chord_key/author_reproduction_early_stopping",
                    "summary_dir": "outputs/mtg_jamendo"}
        for name, relative in expected.items():
            self.assertEqual(values[name], root / relative, name)
        mert, provenance = load_configuration(configs / "experiments/mtg_jamendo/mert.yaml", "mert_segment_extraction")
        self.assertEqual(mert["manifest"], values["manifest"])
        self.assertEqual(mert["output_dir"], values["mert_dir"])
        self.assertEqual(mert["report"], root / "outputs/mtg_jamendo/mert_chord_key/mert_extraction_report.json")
        self.assertEqual(len(provenance["config_files"]), 2)
        self.assertFalse((configs / "experiments/mtg_jamendo/mert_chord_key").exists())

    def test_shared_recipe_has_no_jamendo_paths_targets_or_frequencies(self):
        base_path = default_config_path("training/music2emo.yaml")
        base = yaml.safe_load(base_path.read_text())
        for name in ("dataset", "paths", "task"):
            self.assertNotIn(name, base)
        self.assertNotIn("source", base["features"])
        self.assertNotIn("output_labels", base["model"])
        self.assertNotIn("roc_auc", base["evaluation"])
        mood = yaml.safe_load((base_path.parent / "mood_theme_tagging.yaml").read_text())
        emotion = yaml.safe_load((base_path.parent / "emotion_recognition.yaml").read_text())
        self.assertEqual(mood["training_profile"], "music2emo.yaml")
        self.assertEqual(emotion["training_profile"], "music2emo.yaml")
        self.assertEqual(mood["model"]["output_labels"], 56)
        self.assertEqual(emotion["task"], {"name": "emotion_recognition", "label_type": "binary"})
        self.assertNotIn("paths", emotion)
        self.assertNotIn("model", emotion)

    def test_inherited_values_can_be_overridden_without_mutating_shared_defaults(self):
        preset = default_config_path("experiments/mtg_jamendo/reproduction.yaml")
        with tempfile.TemporaryDirectory() as directory:
            alternate = Path(directory) / "alternate.yaml"
            alternate.write_text(f"training_profile: {preset}\noptimization:\n  epochs: 17\n")
            values, _ = load_configuration(alternate, "jamendo_reproduction")
            self.assertEqual(values["epochs"], 17)
            invalid = Path(directory) / "invalid.yaml"
            invalid.write_text(f"training_profile: {preset}\noptimization:\n  early_stopping:\n    monitor: pr_auc\n")
            with self.assertRaises(ConfigurationError):
                load_configuration(invalid, "jamendo_reproduction")
        values, _ = load_configuration(preset, "jamendo_reproduction")
        self.assertEqual(values["epochs"], 200)

    def test_public_cli_uses_flattened_presets_and_preserves_overrides(self):
        parser = build_parser()
        args = parser.parse_args(["train", "mtg-jamendo", "author-reproduction", "--seed", "64", "--dry-run"])
        self.assertEqual(args.seeds, [64])
        self.assertEqual(args.epochs, 200)
        self.assertEqual(Path(args.yaml_provenance["config_files"][0]["path"]),
                         default_config_path("experiments/mtg_jamendo/reproduction.yaml"))
        args = parser.parse_args(["extract", "mtg-jamendo", "mert", "--dry-run"])
        self.assertEqual(Path(args.yaml_provenance["config_files"][0]["path"]),
                         default_config_path("experiments/mtg_jamendo/mert.yaml"))


if __name__ == "__main__":
    unittest.main()

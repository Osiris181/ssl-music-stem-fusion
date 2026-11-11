"""Synthetic controlled Jamendo training; no source audio or pretrained models."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from stemfusion.cli import build_parser
from stemfusion.configuration import ConfigurationError, default_config_path, load_configuration
from stemfusion.data.jamendo_mid_fusion import SOURCES, JamendoMidFusionDataset, prepare_mid_fusion_data
from stemfusion.data.jamendo_reproduction import JamendoReproductionDataset, prepare_reproduction_data
from stemfusion.experiments._mood_theme_tagging import MID_CONDITIONS as CONDITIONS, MidFusionConfig, run_mid_fusion
from stemfusion.experiments import _mood_theme_tagging as mid_fusion
from stemfusion.experiments.mtg_jamendo import reproduction, train_mid
from stemfusion.features.chords_keys import AUTHOR_COMMIT
from stemfusion.features.mert_segments import MODEL_NAME, MODEL_REVISION
from stemfusion.models.music2emo import Music2EmoClassifier
from stemfusion.models.music2emo_mid_fusion import Music2EmoMidFusion, parameter_count
from stemfusion.training import music2emo
from stemfusion.training.reporting import file_checksum, finish_artifacts, write_json
from stemfusion.training.trainer import set_seed


class JamendoMidFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = MidFusionConfig(self.root / "manifest.jsonl", self.root / "mert", self.root / "harmony",
            self.root / "tags.npy", self.root / "metadata", self.root / "mid", self.root / "summaries",
            epochs=1, batch_size=2, device="cpu", num_workers=0, reference_output_dir=self.root / "reference")
        self.config.author_metadata.mkdir()
        self.tags = [f"mood/theme---tag_{index}" for index in range(56)]
        np.save(self.config.tag_list, np.asarray([f"other---{index}" for index in range(127)] + self.tags))
        write_json(self.config.author_metadata / "chord_root.json", {"N": 0, "C": 1, "D": 3})
        write_json(self.config.author_metadata / "chord_attr.json", {"N": 0, "maj": 1, "min": 5})
        np.save(self.config.author_metadata / "probas_train.npy", np.full(56, .5, dtype=np.float64))
        self.rows = []
        settings = {"model_name": MODEL_NAME, "revision": MODEL_REVISION, "shape": [1, 12, 768],
                    "saved_hidden_states": list(range(1, 13)), "sample_rate": 24000, "segment_seconds": 30.0,
                    "temporal_pooling": "mean", "stem_alignment": "mixture_duration"}
        for index in range(6):
            number = index + 1
            split = ("train", "validation", "test")[index // 2]
            relative = f"{number:02d}/{number}"
            row = {"track_id": f"track_{number:07d}", "split": split, "relative_path": relative + ".mp3",
                   "tags": self.tags[index % 2::2],
                   "paths": {source: str(self.root / "absent_audio" / source / f"{number}.mp3") for source in SOURCES}}
            self.rows.append(row)
            audio = {source: {"path": row["paths"][source], "size": 123, "mtime_ns": 100} for source in SOURCES}
            for source_index, source in enumerate(SOURCES):
                directory = self.config.mert_dir / ("mert_30s" if source == "mixture" else f"mert_30s_{source}") / relative
                directory.mkdir(parents=True)
                for segment in range(2):
                    value = np.zeros((1, 12, 768), dtype=np.float32)
                    value[:, 5] = number / 100 + segment / 10 + source_index
                    value[:, 6] = number / 50 + segment / 10 + source_index
                    np.save(directory / f"segment_{segment}.npy", value)
                metadata = {"track_id": row["track_id"], "split": split, "source": source,
                    "settings": settings, "audio": audio[source], "reference_mixture": audio["mixture"],
                    "segments": [{"file": f"segment_{segment}.npy", "start_sample": segment * 720000,
                                  "end_sample": (segment + 1) * 720000} for segment in range(2)]}
                write_json(directory / "segments.json", metadata)
                finish_artifacts(directory, "fixture", ["segments.json", "segment_0.npy", "segment_1.npy"])
                harmony = self.config.harmony_dir / source / relative
                harmony.mkdir(parents=True)
                (harmony / "normalized_chords.lab").write_text("0 1 C\n1 2 D:min\n")
                (harmony / "key.lab").write_text("None\n" if source == "drums" else "D minor\n")
                write_json(harmony / "metadata.json", {"track_id": row["track_id"], "split": split,
                    "source": source, "audio": audio[source], "extraction": {},
                    "settings": {"author_commit": AUTHOR_COMMIT, "midi_policy": "music2emotion_released",
                                 "normalization": "C_major_A_minor"}})
                finish_artifacts(harmony, "fixture", ["normalized_chords.lab", "key.lab", "metadata.json"])
        self.config.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def tearDown(self):
        self.temporary.cleanup()

    def data(self, sources=SOURCES):
        return prepare_mid_fusion_data(self.config.manifest, self.config.mert_dir, self.config.harmony_dir,
                                        self.config.tag_list, self.config.author_metadata, sources)

    def test_all_sources_keep_baseline_identity_and_do_not_read_audio(self):
        data = self.data()
        baseline = prepare_reproduction_data(self.config.manifest, self.config.mert_dir, self.config.harmony_dir,
                                              self.config.tag_list, self.config.author_metadata)
        self.assertEqual(data.mixture.data_fingerprint, baseline.data_fingerprint)
        self.assertEqual(data.audit["split_counts"], {"train": 2, "validation": 2, "test": 2})
        self.assertEqual(data.audit["unavailable_keys_mapped_to_major_as_author"]["drums"], 6)
        self.assertEqual(data.splits["train"][0].views["drums"].mode, 0)
        self.assertEqual(set(data.splits["train"][0].views), set(SOURCES))
        self.assertFalse((self.root / "absent_audio").exists())
        subset = self.data(SOURCES[:4])
        self.assertEqual(data.condition(SOURCES[:4]).data_fingerprint, subset.condition(SOURCES[:4]).data_fingerprint)
        self.assertEqual(data.cohort_fingerprint, subset.cohort_fingerprint)

    def test_training_uses_one_shared_temporal_sample_and_eval_means_all_segments(self):
        data = self.data()
        records = data.splits["train"]
        np.random.seed(32)
        baseline = JamendoReproductionDataset(data.mixture.splits["train"], training=True)[0]
        np.random.seed(32)
        sample = JamendoMidFusionDataset(records, sources=SOURCES, training=True)[0]
        torch.testing.assert_close(sample["x_mert"][0], baseline["x_mert"], rtol=0, atol=0)
        for index in range(6):
            torch.testing.assert_close(sample["x_mert"][index], sample["x_mert"][0] + index)
        evaluation = JamendoMidFusionDataset(records, sources=SOURCES, training=False)[0]
        np.testing.assert_array_equal(evaluation["x_mert"][0].numpy(), records[0].views["mixture"].segments.mean(0))

    def test_invalid_source_cache_and_alignment_are_rejected(self):
        directory = self.config.mert_dir / "mert_30s_guitar/01/1"
        metadata = json.loads((directory / "segments.json").read_text())
        metadata["segments"][0]["end_sample"] -= 1
        write_json(directory / "segments.json", metadata)
        finish_artifacts(directory, "fixture", ["segments.json", "segment_0.npy", "segment_1.npy"])
        with self.assertRaisesRegex(ValueError, "boundaries differ"):
            self.data()
        (directory / "segment_0.npy").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Changed or incomplete"):
            self.data()

    def test_wrong_source_audio_identity_is_rejected(self):
        directory = self.config.harmony_dir / "guitar/01/1"
        metadata = json.loads((directory / "metadata.json").read_text())
        metadata["audio"]["path"] = self.rows[0]["paths"]["mixture"]
        write_json(directory / "metadata.json", metadata)
        finish_artifacts(directory, "fixture", ["normalized_chords.lab", "key.lab", "metadata.json"])
        with self.assertRaisesRegex(ValueError, "audio path differs"):
            self.data()

    def test_single_input_model_is_exactly_the_original_model(self):
        set_seed(32)
        original = Music2EmoClassifier().eval()
        set_seed(32)
        current = Music2EmoMidFusion(1).eval()
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, current.state_dict()[name], rtol=0, atol=0)
        data = self.data(("mixture",))
        sample = JamendoMidFusionDataset(data.splits["train"], sources=("mixture",), training=False)[0]
        values = {name: value.unsqueeze(0) for name, value in sample.items() if name.startswith("x_")}
        with torch.inference_mode():
            torch.testing.assert_close(current(values), original({name: value[:, 0] for name, value in values.items()}), rtol=0, atol=0)

    def test_controls_match_full_parameter_count_and_repeat_identical_encoded_vectors(self):
        for count in (4, 5, 6):
            fused = Music2EmoMidFusion(count)
            control = Music2EmoMidFusion(count, repeated_mixture=True)
            self.assertEqual(sum(p.numel() for p in fused.parameters()), parameter_count(count))
            self.assertEqual(sum(p.numel() for p in control.parameters()), parameter_count(count))
        sample = JamendoMidFusionDataset(self.data(("mixture",)).splits["train"], sources=("mixture",), training=False)[0]
        values = {name: value.unsqueeze(0) for name, value in sample.items() if name.startswith("x_")}
        control.train()
        captured = []
        handle = control.input_proj[0].register_forward_pre_hook(lambda module, arguments: captured.append(arguments[0]))
        output = control(values)
        handle.remove()
        chunks = captured[0].split(1545, dim=1)
        self.assertTrue(all(torch.equal(chunks[0], chunk) for chunk in chunks[1:]))
        output.sum().backward()
        self.assertGreater(float(control.chord_root_embedding.weight.grad.abs().sum()), 0)

    def test_threshold_selection_uses_validation_only_and_labelwise_accuracy(self):
        validation = {"probabilities": np.asarray([[.9, .1], [.1, .9]]), "targets": np.asarray([[1, 0], [0, 1]])}
        threshold = music2emo.select_threshold(validation)
        self.assertEqual(threshold, .11)
        self.assertEqual(music2emo.threshold_metrics(validation, threshold)["f1_macro"], 1.)
        test = {"probabilities": np.asarray([[.9, .9], [.1, .9]]), "targets": validation["targets"]}
        metrics = music2emo.threshold_metrics(test, threshold)
        self.assertEqual(metrics["accuracy"], .75)
        self.assertEqual(metrics["subset_accuracy"], .5)
        self.assertEqual(music2emo.select_threshold(validation), threshold)

    def test_stopping_and_epoch_seed_match_retained_reproduction(self):
        first = {"best_score": float("inf"), "wait_count": 0, "stopped_epoch": None}
        second = first.copy()
        for epoch, loss in enumerate((1., .9, .89999, .89998), 1):
            for function, state in ((music2emo._update_early_stopping, first), (reproduction._update_early_stopping, second)):
                function(state, loss, epoch, patience=2, min_delta=1e-4)
            self.assertEqual(first, second)
            self.assertEqual(music2emo.epoch_seed(32, epoch), reproduction.epoch_seed(32, epoch))

    def test_dry_run_and_validation_only_do_not_train_or_write_outputs(self):
        with patch("stemfusion.experiments._mood_theme_tagging.prepare_cached_fusion_data", side_effect=AssertionError("No feature access")):
            self.assertTrue(run_mid_fusion(replace(self.config, dry_run=True))["dry_run"])
        with patch("stemfusion.training.music2emo.fit_model", side_effect=AssertionError("No fitting")), patch.object(reproduction, "fit_seed", side_effect=AssertionError("No fitting")):
            self.assertTrue(run_mid_fusion(replace(self.config, validate_only=True))["validate_only"])
        self.assertFalse(self.config.output_dir.exists())
        self.assertFalse(self.config.reference_output_dir.exists())
        self.assertFalse(self.config.summary_dir.exists())

    def test_full_five_seed_matrix_saves_paired_results_and_reuses_all_checkpoints(self):
        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_mid_fusion(self.config)
        self.assertEqual(set(summary["conditions"]), set(CONDITIONS))
        self.assertEqual(summary["seeds"], [32, 64, 128, 256, 512])
        self.assertEqual(len(summary["paired_differences"]), 3)
        self.assertFalse(summary["significance_tests_performed"])
        self.assertTrue((self.config.summary_dir / "summary_mid_fusion.txt").is_file())
        for condition, runs in summary["runs"].items():
            self.assertEqual(len(runs), 5)
            for seed, result in runs.items():
                self.assertTrue(Path(result["checkpoint"]).is_file())
                folder = self.config.output_dir / condition / f"seed_{seed}"
                self.assertTrue((folder / "run_manifest.json").is_file())
                self.assertTrue((folder / "predictions_test.npz").is_file())
                self.assertTrue((folder / "predictions_validation.npz").is_file())
        for name, paired in summary["paired_differences"].items():
            metric = "pr_auc_macro"
            expected = [summary["runs"][name][seed]["test"][metric] - summary["runs"][paired["control"]][seed]["test"][metric]
                        for seed in self.config.seeds]
            np.testing.assert_array_equal(paired["metrics"][metric]["values"], expected)
        values = summary["conditions"]["mixture_only"]["pr_auc_macro"]["values"]
        self.assertAlmostEqual(summary["conditions"]["mixture_only"]["pr_auc_macro"]["sample_sd"], np.std(values, ddof=1))
        before = {path: file_checksum(path) for path in self.config.reference_output_dir.rglob("*") if path.is_file()}
        with patch.object(reproduction, "Music2EmoClassifier", side_effect=AssertionError("No baseline retraining")), patch.object(music2emo, "set_seed", side_effect=AssertionError("No retraining")), contextlib.redirect_stdout(io.StringIO()):
            again = run_mid_fusion(self.config)
        self.assertEqual(summary["conditions"], again["conditions"])
        self.assertTrue(all(file_checksum(path) == checksum for path, checksum in before.items()))
        full_summary = file_checksum(self.config.summary_dir / "summary_mid_fusion.json")
        with contextlib.redirect_stdout(io.StringIO()):
            partial = run_mid_fusion(replace(self.config, seeds=(32,), selected_run="mixture_only"))
        self.assertEqual(partial["seeds"], [32])
        self.assertEqual(file_checksum(self.config.summary_dir / "summary_mid_fusion.json"), full_summary)
        self.assertTrue((self.config.summary_dir / "summary_mid_fusion_run_mixture_only_seeds_32.txt").is_file())
        with self.assertRaisesRegex(ValueError, "incompatible|mismatch"):
            run_mid_fusion(replace(self.config, epochs=2, selected_run="mixture_only", seeds=(32,)))

    def test_interrupted_fusion_resume_is_bit_exact(self):
        data = self.data(SOURCES[:4]).condition(SOURCES[:4])
        config = replace(self.config, epochs=2, seeds=(32,))
        def fit(selected, callback=None):
            return music2emo.fit_model(selected, data, 32, model_factory=lambda: Music2EmoMidFusion(4),
                dataset_factory=lambda records, training: JamendoMidFusionDataset(records, sources=SOURCES[:4], training=training),
                implementation="fixture", protocol_updates={"condition": "fixture"}, epoch_callback=callback)
        def stop(epoch):
            raise RuntimeError("synthetic interruption")
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            fit(config, stop)
        resumed = fit(config)
        uninterrupted_config = replace(config, output_dir=self.root / "uninterrupted")
        uninterrupted = fit(uninterrupted_config)
        self.assertEqual(resumed["best_epoch"], uninterrupted["best_epoch"])
        first = np.load(config.output_dir / "seed_32/predictions_test.npz")
        second = np.load(uninterrupted_config.output_dir / "seed_32/predictions_test.npz")
        np.testing.assert_array_equal(first["logits"], second["logits"])
        self.assertEqual(json.loads((config.output_dir / "seed_32/history.json").read_text()),
                         json.loads((uninterrupted_config.output_dir / "seed_32/history.json").read_text()))
        first.close()
        second.close()

    def test_yaml_inheritance_cli_overrides_and_direct_entry_point(self):
        preset = default_config_path("experiments/mtg_jamendo/mid_fusion.yaml")
        defaults, provenance = load_configuration(preset, "jamendo_mid_fusion")
        self.assertEqual(defaults["runs"], CONDITIONS)
        self.assertEqual(defaults["epochs"], 200)
        self.assertEqual(defaults["early_stopping_patience"], 10)
        self.assertTrue(defaults["verify_checksums"])
        self.assertEqual(len(provenance["config_files"]), 3)
        parser = build_parser()
        args = parser.parse_args(["train", "mtg-jamendo", "mid-fusion", "--run", "repeated_mixture_4", "--seed", "64", "--dry-run"])
        self.assertEqual((args.run, args.seeds), ("repeated_mixture_4", [64]))
        self.assertTrue(args.verify_checksums)
        args = parser.parse_args(["train", "mtg-jamendo", "mid-fusion", "--no-verify-checksums", "--dry-run"])
        self.assertFalse(args.verify_checksums)
        alternate = self.root / "invalid.yaml"
        alternate.write_text(f"training_profile: {preset}\nruntime:\n  verify_checksums: false\n")
        self.assertFalse(load_configuration(alternate, "jamendo_mid_fusion")[0]["verify_checksums"])
        args = parser.parse_args(["train", "mtg-jamendo", "mid-fusion", "--config", str(alternate), "--verify-checksums"])
        self.assertTrue(args.verify_checksums)
        alternate.write_text(f"training_profile: {preset}\nruntime:\n  verify_checksums: 'false'\n")
        with self.assertRaises(ConfigurationError):
            load_configuration(alternate, "jamendo_mid_fusion")
        old_settings = json.loads(json.dumps(provenance["yaml_settings_before_cli_overrides"]))
        old_settings["runtime"].pop("verify_checksums")
        alternate.write_text(yaml.safe_dump(old_settings))
        self.assertTrue(load_configuration(alternate, "jamendo_mid_fusion")[0]["verify_checksums"])
        alternate.write_text(f"training_profile: {preset}\nfeatures:\n  harmony_encoder: independent\n")
        with self.assertRaises(ConfigurationError):
            load_configuration(alternate, "jamendo_mid_fusion")
        script = Path(train_mid.__file__)
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        process = subprocess.run([sys.executable, str(script), "--dry-run", "--run", "mixture_only", "--no-verify-checksums"],
                                 cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("no features/model loaded", process.stderr)
        self.assertIn('"verify_checksums": false', process.stderr)
        with patch("stemfusion.experiments._mood_theme_tagging.run_mid_fusion") as run:
            self.assertEqual(train_mid.main(["--output-dir", str(self.root / "isolated"), "--dry-run"]), 0)
            config = run.call_args.args[0]
            self.assertEqual(config.summary_dir, self.root / "isolated")
            self.assertEqual(config.reference_output_dir, self.root / "isolated/reference")

    def test_optional_checksums_validate_without_training_or_writing(self):
        config = replace(self.config, validate_only=True, verify_checksums=False)
        with patch.object(music2emo, "fit_model", side_effect=AssertionError("No fitting")), patch.object(
                reproduction, "fit_seed", side_effect=AssertionError("No fitting")), self.assertLogs(
                mid_fusion.LOGGER, level="WARNING") as messages:
            result = run_mid_fusion(config)
        self.assertIn("may go undetected", messages.output[0])
        self.assertFalse(result["verify_checksums"])
        self.assertFalse(result["audit"]["cache_verification"]["feature_payload_checksums_verified"])
        self.assertTrue(result["audit"]["source_specific_harmony"])
        self.assertFalse(config.output_dir.exists())
        self.assertFalse(config.summary_dir.exists())
        with self.assertRaisesRegex(ValueError, "boolean"):
            run_mid_fusion(replace(config, verify_checksums="false"))

    def test_optional_checksums_preserve_historical_checkpoints_and_results(self):
        conditions = {name: CONDITIONS[name] for name in (
            "mixture_only", "repeated_mixture_4", "mixture_guitar_drums_piano")}
        config = replace(self.config, seeds=(32,), conditions=conditions)
        # Model historical caches with their original full loader/audit and the
        # independently recorded, pre-option implementation identity.
        historical_identity = "ba940df5a28bb5ed138adfa3365e35db670a9b3c1719a47a30630c241515a544"
        with patch.object(mid_fusion, "prepare_cached_fusion_data", side_effect=lambda *args, **kwargs: self.data(SOURCES[:4])), patch.object(
                mid_fusion, "implementation_identity", return_value=historical_identity), contextlib.redirect_stdout(io.StringIO()):
            historical = run_mid_fusion(config)
        self.assertEqual(mid_fusion.implementation_identity(), historical_identity)
        artifacts = {path: file_checksum(path) for root in (config.output_dir, config.reference_output_dir)
                     for path in root.rglob("*") if path.is_file() and path.suffix in {".json", ".npz"}}
        for verify in (False, True):
            with patch.object(reproduction, "Music2EmoClassifier", side_effect=AssertionError("No baseline retraining")), patch.object(
                    music2emo, "set_seed", side_effect=AssertionError("No fusion retraining")), contextlib.redirect_stdout(io.StringIO()):
                current = run_mid_fusion(replace(config, verify_checksums=verify))
            self.assertEqual(historical["conditions"], current["conditions"])
            self.assertEqual(current["data_audit"]["cache_verification"]["feature_payload_checksums_verified"], verify)
            self.assertEqual(current["execution_implementation_sha256"],
                             mid_fusion.execution_identity("mid"))
            self.assertNotIn("cache_verification", current["runs"]["mixture_guitar_drums_piano"][32]["data_audit"])
            self.assertTrue(all(file_checksum(path) == expected for path, expected in artifacts.items()))

    def test_optional_checksums_preserve_exact_fusion_training_predictions(self):
        config = replace(self.config, seeds=(32,), selected_run="mixture_guitar_drums_piano")
        with contextlib.redirect_stdout(io.StringIO()):
            verified = run_mid_fusion(config)
            trusted = run_mid_fusion(replace(config, verify_checksums=False, output_dir=self.root / "trusted_mid"))
        self.assertEqual(verified["conditions"], trusted["conditions"])
        first = self.config.output_dir / config.selected_run / "seed_32/predictions_test.npz"
        second = self.root / "trusted_mid" / config.selected_run / "seed_32/predictions_test.npz"
        with np.load(first) as a, np.load(second) as b:
            np.testing.assert_array_equal(a["logits"], b["logits"])
        self.assertFalse(trusted["runs"][config.selected_run][32]["data_audit"]["cache_verification"]["feature_payload_checksums_verified"])

    def test_optional_checksums_keep_structural_checks_mandatory(self):
        config = replace(self.config, seeds=(32,), selected_run="mixture_guitar_drums_piano",
                         verify_checksums=False, validate_only=True)
        path = config.mert_dir / "mert_30s_guitar/01/1/segment_0.npy"
        original = path.read_bytes()
        for value in (np.full((1, 12, 768), np.nan, dtype=np.float32), np.zeros((12, 768), dtype=np.float32)):
            np.save(path, value)
            with self.assertRaisesRegex(ValueError, "finite float32"):
                run_mid_fusion(config)
        path.write_bytes(original)
        metadata_path = path.parent / "segments.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["segments"][0]["end_sample"] = 100
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "segment boundaries differ"):
            run_mid_fusion(config)
        self.assertFalse(config.output_dir.exists())


if __name__ == "__main__":
    unittest.main()

"""Binary PMEmo targets through the shared Jamendo fitting/sampling workflow."""

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
from stemfusion.data.emotion_mid_fusion import prepare_emotion_features
from stemfusion.data.jamendo_mid_fusion import SOURCES, JamendoMidFusionDataset
from stemfusion.experiments import _emotion_recognition as controller
from stemfusion.experiments.pmemo import train_mid
from stemfusion.features.chords_keys import AUTHOR_COMMIT
from stemfusion.features.mert_segments import MODEL_NAME, MODEL_REVISION
from stemfusion.models.music2emo_emotion import Music2EmoEmotionMidFusion, parameter_count
from stemfusion.training import music2emo
from stemfusion.training.emotion import BinaryEmotionTask
from stemfusion.training.reporting import file_checksum, finish_artifacts, write_json


class TinyFusion(torch.nn.Module):
    def __init__(self, inputs, *, repeated_mixture=False):
        super().__init__()
        self.inputs, self.repeated = inputs, repeated_mixture
        self.output = torch.nn.Linear(inputs * 1536, 1)

    def forward(self, values):
        inputs = values["x_mert"].flatten(1)
        if self.repeated:
            inputs = inputs.repeat(1, self.inputs)
        return self.output(inputs)


class PMEmoMidFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = controller.EmotionMidFusionConfig(self.root / "manifest.jsonl", self.root / "mert",
            self.root / "harmony", self.root / "root.json", self.root / "attr.json", self.root / "runs", self.root / "summaries",
            epochs=1, batch_size=2, device="cpu", num_workers=0)
        write_json(self.config.chord_root, {"N": 0, "C": 1, "D": 3})
        write_json(self.config.chord_attr, {"N": 0, "maj": 1, "min": 5})
        self.rows = []
        settings = {"model_name": MODEL_NAME, "revision": MODEL_REVISION, "shape": [1, 12, 768],
                    "saved_hidden_states": list(range(1, 13)), "sample_rate": 24000, "segment_seconds": 30.0,
                    "temporal_pooling": "mean", "tail_policy": "retain", "processor_normalize": True,
                    "minimum_input_samples": 400, "short_segment_policy": "pad_to_minimum", "stem_alignment": "mixture_duration"}
        for index in range(6):
            track_id = str(index + 1)
            split = ("train", "validation", "test")[index // 2]
            row = {"dataset": "pmemo", "track_id": track_id, "relative_path": track_id + ".mp3", "split": split,
                   "annotations": {"valence": 0.5 if index % 2 else 0.25, "arousal": 0.25 if index % 2 else 0.75},
                   "paths": {source: str(self.root / "absent_audio" / source / f"{track_id}.mp3") for source in SOURCES}}
            self.rows.append(row)
            audio = {source: {"path": row["paths"][source], "size": 123, "mtime_ns": 100} for source in SOURCES}
            for source_index, source in enumerate(SOURCES):
                directory = self.config.mert_dir / ("mert_30s" if source == "mixture" else f"mert_30s_{source}") / track_id
                directory.mkdir(parents=True)
                for segment in range(2):
                    values = np.zeros((1, 12, 768), dtype=np.float32)
                    values[:, 5] = (index + 1) / 100 + segment / 10 + source_index
                    values[:, 6] = (index + 1) / 50 + segment / 10 + source_index
                    np.save(directory / f"segment_{segment}.npy", values)
                write_json(directory / "segments.json", {"track_id": track_id, "split": split, "source": source,
                    "settings": settings, "audio": audio[source], "reference_mixture": audio["mixture"],
                    "segments": [{"file": f"segment_{segment}.npy", "start_sample": segment * 720000,
                                  "end_sample": (segment + 1) * 720000} for segment in range(2)]})
                finish_artifacts(directory, "fixture", ["segments.json", "segment_0.npy", "segment_1.npy"])
                harmony = self.config.harmony_dir / source / track_id
                harmony.mkdir(parents=True)
                (harmony / "normalized_chords.lab").write_text("0 1 C\n1 2 D:min\n")
                (harmony / "key.lab").write_text("None\n" if source == "drums" else "D minor\n")
                write_json(harmony / "metadata.json", {"track_id": track_id, "split": split, "source": source,
                    "audio": audio[source], "settings": {"author_commit": AUTHOR_COMMIT,
                    "midi_policy": "music2emotion_released", "normalization": "C_major_A_minor"}})
                finish_artifacts(harmony, "fixture", ["metadata.json", "normalized_chords.lab", "key.lab"])
        self.write_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self):
        self.config.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def features(self, **kwargs):
        return prepare_emotion_features(self.config.manifest, self.config.mert_dir, self.config.harmony_dir,
                                        self.config.chord_root, self.config.chord_attr, **kwargs)

    def test_fixed_midpoint_labels_and_independent_target_identities(self):
        features = self.features()
        valence, arousal = features.target("valence"), features.target("arousal")
        self.assertEqual([int(record.labels[0]) for record in valence.splits["train"]], [0, 1])
        self.assertEqual([int(record.labels[0]) for record in arousal.splits["train"]], [1, 0])
        self.assertNotEqual(valence.cohort_fingerprint, arousal.cohort_fingerprint)
        self.assertEqual(valence.mixture.prevalence.tolist(), [0.5])
        self.assertEqual(valence.audit["class_counts"]["test"], {"low": 1, "high": 1})
        np.testing.assert_array_equal(valence.splits["train"][0].views["mixture"].segments,
                                      arousal.splits["train"][0].views["mixture"].segments)
        self.assertIs(valence.splits["train"][0].views["mixture"].segments,
                      arousal.splits["train"][0].views["mixture"].segments)
        self.assertFalse((self.root / "absent_audio").exists())

    def test_loss_prevalence_uses_only_training_labels(self):
        features = self.features()
        # Add an in-memory low training example; validation/test stay balanced.
        features.annotations["7"] = dict(features.annotations["1"])
        features.splits["train"].append(replace(features.splits["train"][0], track_id="7"))
        data = features.target("valence")
        self.assertEqual(data.mixture.prevalence.tolist(), [1 / 3])
        self.assertEqual(data.mixture.prevalence.dtype, np.float64)

    def test_shared_temporal_sampling_and_source_specific_harmony(self):
        data = self.features().target("valence")
        np.random.seed(32)
        sample = JamendoMidFusionDataset(data.splits["train"], sources=SOURCES, training=True)[0]
        for index in range(6):
            torch.testing.assert_close(sample["x_mert"][index], sample["x_mert"][0] + index)
        self.assertEqual(sample["x_key"][2].item(), 0)  # Unavailable drums key -> released major fallback.
        self.assertEqual(sample["x_key"][0].item(), 1)
        evaluation = JamendoMidFusionDataset(data.splits["train"], sources=SOURCES, training=False)[0]
        np.testing.assert_array_equal(evaluation["x_mert"][0], data.splits["train"][0].views["mixture"].segments.mean(0))

    def test_optional_checksums_preserve_feature_and_cohort_identities(self):
        verified = self.features()
        trusted = self.features(verify_checksums=False)
        for target in ("valence", "arousal"):
            a, b = verified.target(target), trusted.target(target)
            self.assertEqual(a.cohort_fingerprint, b.cohort_fingerprint)
            self.assertEqual(a.condition(SOURCES).data_fingerprint, b.condition(SOURCES).data_fingerprint)
            self.assertFalse(b.audit["cache_verification"]["feature_payload_checksums_verified"])

    def test_corrupted_and_misaligned_cache_payloads_are_rejected(self):
        path = self.config.mert_dir / "mert_30s_guitar/1/segment_0.npy"
        np.save(path, np.full((1, 12, 768), np.nan, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "Changed or incomplete"):
            self.features()
        with self.assertRaisesRegex(ValueError, "finite float32"):
            self.features(verify_checksums=False)

    def test_unverified_mode_still_rejects_alignment_and_wrong_audio_identity(self):
        path = self.config.mert_dir / "mert_30s_guitar/1/segments.json"
        metadata = json.loads(path.read_text())
        metadata["segments"][0]["end_sample"] -= 1
        write_json(path, metadata)
        with self.assertRaisesRegex(ValueError, "boundaries differ"):
            self.features(verify_checksums=False)

    def test_degenerate_split_is_not_silently_dropped(self):
        self.rows[0]["annotations"]["valence"] = 0.5
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Both low/high"):
            self.features().target("valence")

    def test_binary_metrics_average_both_classes_and_threshold_ties_choose_lowest(self):
        task = BinaryEmotionTask("valence")
        arrays = {"targets": np.asarray([[0], [0], [1], [1]]), "probabilities": np.full((4, 1), 0.5)}
        self.assertAlmostEqual(task.threshold_metrics(arrays, 0.5)["f1_macro"], 1 / 3)
        self.assertEqual(task.select_threshold(arrays), 0.0)
        arrays = {"targets": np.asarray([[0], [0], [0], [1]]),
                  "probabilities": np.asarray([[0.1], [0.2], [0.9], [0.8]], dtype=np.float32)}
        metrics = task.ranking_metrics(torch.from_numpy(arrays["probabilities"]), torch.from_numpy(arrays["targets"]))
        self.assertAlmostEqual(metrics["pr_auc_macro"], np.mean(metrics["per_class_average_precision"]))
        self.assertNotEqual(metrics["pr_auc_macro"], metrics["per_class_average_precision"][1])

    def test_binary_model_shapes_parameter_matching_and_repeated_vectors(self):
        data = self.features().target("valence")
        for count in (1, 4, 5, 6):
            model = Music2EmoEmotionMidFusion(count).eval()
            selected = SOURCES[:count]
            sample = JamendoMidFusionDataset(data.splits["train"], sources=selected, training=False)[0]
            values = {name: value.unsqueeze(0) for name, value in sample.items() if name.startswith("x_")}
            with torch.inference_mode():
                self.assertEqual(model(values).shape, (1, 1))
            self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), parameter_count(count))
            if count > 1:
                control = Music2EmoEmotionMidFusion(count, repeated_mixture=True).train()
                self.assertEqual(sum(parameter.numel() for parameter in control.parameters()), parameter_count(count))
                encoded = []
                handle = control.input_proj.register_forward_pre_hook(lambda module, args: encoded.append(args[0].detach()))
                mixture = {name: value[:, :1] for name, value in values.items()}
                control(mixture)
                handle.remove()
                for block in encoded[0].split(1545, dim=1)[1:]:
                    torch.testing.assert_close(block, encoded[0][:, :1545], rtol=0, atol=0)

    def test_full_five_seed_matrix_two_targets_reuses_all_completed_models(self):
        features = self.features()
        with patch.object(controller, "Music2EmoEmotionMidFusion", TinyFusion), patch.object(
                controller, "parameter_count", side_effect=lambda count: count * 1536 + 1), contextlib.redirect_stdout(io.StringIO()):
            result = controller.run_mid_fusion(self.config, features=features)
            artifacts = {path: file_checksum(path) for path in self.config.output_dir.rglob('*') if path.is_file()}
            with patch.object(music2emo, "set_seed", side_effect=AssertionError("Completed runs must not retrain")):
                reused = controller.run_mid_fusion(self.config, features=features)
        self.assertEqual(set(result["targets"]), {"valence", "arousal"})
        self.assertEqual(len(list(self.config.output_dir.rglob("best.pt"))), 70)
        self.assertTrue(all(file_checksum(path) == digest for path, digest in artifacts.items()))
        for target in ("valence", "arousal"):
            summary = result["targets"][target]
            self.assertEqual(summary["conditions"], reused["targets"][target]["conditions"])
            self.assertEqual(len(summary["conditions"]), 7)
            self.assertEqual(summary["seeds"], [32, 64, 128, 256, 512])
            values = summary["conditions"]["mixture_only"]["accuracy"]["values"]
            self.assertAlmostEqual(summary["conditions"]["mixture_only"]["accuracy"]["sample_sd"], np.std(values, ddof=1))
            self.assertTrue((self.config.summary_dir / f"summary_mid_fusion_{target}.txt").is_file())
            for row in summary["runs"].values():
                for fit in row.values():
                    self.assertEqual(fit["protocol"]["target"], target)
                    self.assertEqual(fit["protocol"]["output_labels"], 1)

    def test_interrupted_binary_fit_resume_is_bit_exact_and_changed_settings_are_rejected(self):
        data = self.features().target("valence").condition(("mixture",))
        config = replace(self.config, epochs=2, targets=("valence",), seeds=(32,))
        def fit(selected, callback=None):
            return music2emo.fit_model(selected, data, 32, model_factory=lambda: TinyFusion(1),
                dataset_factory=lambda records, training: JamendoMidFusionDataset(records, sources=("mixture",), training=training),
                implementation="fixture", task=BinaryEmotionTask("valence"),
                protocol_updates={"condition": "fixture"}, epoch_callback=callback)
        def stop(epoch):
            raise RuntimeError("synthetic interruption")
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            fit(config, stop)
        resumed = fit(config)
        full = fit(replace(config, output_dir=self.root / "uninterrupted"))
        self.assertEqual(resumed["best_epoch"], full["best_epoch"])
        with np.load(config.output_dir / "seed_32/predictions_test.npz") as a, np.load(self.root / "uninterrupted/seed_32/predictions_test.npz") as b:
            np.testing.assert_array_equal(a["logits"], b["logits"])
        with self.assertRaisesRegex(ValueError, "incompatible"):
            fit(replace(config, learning_rate=2e-4))

    def test_actual_binary_architecture_fits_and_saves_single_logit_predictions(self):
        config = replace(self.config, targets=("valence",), seeds=(32,), selected_run="mixture_only")
        with contextlib.redirect_stdout(io.StringIO()):
            result = controller.run_mid_fusion(config)
        directory = config.output_dir / "valence/mixture_only/seed_32"
        saved = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(saved["model_state"]["output_proj.2.weight"].shape, (1, 256))
        with np.load(directory / "predictions_test.npz") as predictions:
            self.assertEqual(predictions["logits"].shape, (2, 1))
            self.assertEqual(predictions["targets"].shape, (2, 1))
            self.assertTrue(np.isfinite(predictions["probabilities"]).all())
        fit = result["targets"]["valence"]["runs"]["mixture_only"][32]
        self.assertEqual(fit["trainable_parameters"], parameter_count(1))
        self.assertEqual(fit["best_epoch"], 1)

    def test_validate_only_never_fits_or_writes(self):
        with patch.object(music2emo, "fit_model", side_effect=AssertionError("No training")), contextlib.redirect_stdout(io.StringIO()):
            result = controller.run_mid_fusion(replace(self.config, validate_only=True))
        self.assertTrue(result["validate_only"])
        self.assertEqual(set(result["targets"]), {"valence", "arousal"})
        self.assertFalse(self.config.output_dir.exists())
        self.assertFalse(self.config.summary_dir.exists())

    def test_yaml_defaults_cli_overrides_and_jamendo_shared_hyperparameters(self):
        preset = default_config_path("experiments/pmemo/mid_fusion.yaml")
        defaults, provenance = load_configuration(preset, "emotion_mid_fusion")
        reference, _ = load_configuration(default_config_path("experiments/mtg_jamendo/mid_fusion.yaml"), "jamendo_mid_fusion")
        for key in ("epochs", "batch_size", "learning_rate", "early_stopping_patience", "early_stopping_min_delta", "seeds", "device", "num_workers"):
            self.assertEqual(defaults[key], reference[key], key)
        self.assertEqual(defaults["runs"], reference["runs"])
        self.assertTrue(defaults["resume"])
        self.assertTrue(defaults["verify_checksums"])
        args = build_parser().parse_args(["train", "pmemo", "mid-fusion", "--target", "valence", "--seed", "64",
                                         "--run", "mixture_only", "--no-verify-checksums", "--dry-run"])
        config = controller.config_from_args(args)
        self.assertEqual((config.targets, config.seeds, config.selected_run), (("valence",), (64,), "mixture_only"))
        self.assertFalse(config.verify_checksums)
        self.assertEqual(len(provenance["config_files"]), 3)
        invalid = self.root / "invalid.yaml"
        invalid.write_text(f"training_profile: {preset}\ntask:\n  label_threshold: 0.6\n")
        with self.assertRaises(ConfigurationError):
            load_configuration(invalid, "emotion_mid_fusion")
        invalid.write_text(f"training_profile: {preset}\nmodel:\n  output_labels: 56\n")
        with self.assertRaises(ConfigurationError):
            load_configuration(invalid, "emotion_mid_fusion")

    def test_direct_entry_outside_repository_and_output_override(self):
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        process = subprocess.run([sys.executable, str(Path(train_mid.__file__)), "--dry-run", "--target", "valence"],
            cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["targets"], ["valence"])
        with patch.object(controller, "run_mid_fusion") as run:
            self.assertEqual(train_mid.main(["--output-dir", str(self.root / "isolated"), "--dry-run"]), 0)
        self.assertEqual(run.call_args.args[0].summary_dir, self.root / "isolated")


if __name__ == "__main__":
    unittest.main()

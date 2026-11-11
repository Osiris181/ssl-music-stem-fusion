from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from stemfusion.configuration import ConfigurationError, default_config_path, load_configuration
from stemfusion.features.mert_segments import (
    CONV_KERNELS, CONV_STRIDES, MODEL_FILES, MERTSegmentConfig, _MERTRuntime,
    extract_mert_segments, minimum_conv_samples, pool_hidden_states, prepare_mert_segment,
)


class MERTSegmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audio = self.root / "audio.wav"
        self.audio.write_bytes(b"synthetic audio identity")
        self.manifest = self.root / "manifest.jsonl"
        self.row = {"track_id": "track_0000001", "split": "train", "relative_path": "01/1.mp3",
                    "tags": ["mood/theme---happy"], "paths": {"mixture": str(self.audio), "guitar": str(self.audio)}}
        self.manifest.write_text(json.dumps(self.row) + "\n")
        self.model_root = self.root / "model"
        self.model_root.mkdir()
        for name in MODEL_FILES:
            (self.model_root / name).write_bytes(b"synthetic model asset")
        self.config = MERTSegmentConfig(self.manifest, self.root / "features", model_path=self.model_root,
                                       device="cpu", splits=("train",))

    def tearDown(self):
        self.temp.cleanup()

    @property
    def folder(self):
        return self.config.output_dir / "mert_30s/01/1"

    @staticmethod
    def extractor(path):
        yield 0, 720000, np.ones((1, 12, 768), dtype=np.float32)
        yield 720000, 720001, np.full((1, 12, 768), 2, dtype=np.float32)

    def run_fixture(self, config=None, sources=("mixture",), extractor=None):
        return extract_mert_segments(config or self.config, sources,
                                     extractor_factory=lambda *_: extractor or self.extractor)

    def test_minimum_comes_from_convolutions_and_padding_is_minimal(self):
        self.assertEqual(minimum_conv_samples(CONV_KERNELS, CONV_STRIDES), 400)
        for size in (1, 399, 400, 720, 720000):
            original = torch.arange(size, dtype=torch.float32)
            padded, count = prepare_mert_segment(original, "pad_to_minimum")
            self.assertEqual(count, max(0, 400 - size))
            self.assertEqual(len(padded), max(size, 400))
            torch.testing.assert_close(padded[:size], original, rtol=0, atol=0)
            if size >= 400:
                self.assertIs(padded, original)
            else:
                self.assertEqual(padded[size:].count_nonzero(), 0)
        with self.assertRaisesRegex(ValueError, "400"):
            prepare_mert_segment(torch.ones(399), "error")
        for invalid in (torch.empty(0), torch.ones(1, 400), torch.tensor([float("nan")])):
            with self.assertRaises(ValueError):
                prepare_mert_segment(invalid, "pad_to_minimum")

    def test_pooling_matches_author_and_handles_single_frame(self):
        generator = torch.Generator().manual_seed(32)
        states = tuple(torch.randn(1, 7, 768, generator=generator) for _ in range(13))
        author = torch.stack(states).squeeze()[1:, :, :].unsqueeze(0).mean(dim=2)
        torch.testing.assert_close(pool_hidden_states(states), author, rtol=0, atol=0)
        single = tuple(torch.full((1, 1, 768), float(i)) for i in range(13))
        self.assertEqual(tuple(pool_hidden_states(single).shape), (1, 12, 768))
        torch.testing.assert_close(pool_hidden_states(single)[0, :, 0], torch.arange(1, 13).float())
        for invalid in (states[:-1], tuple(torch.ones(1, 2, 1024) for _ in range(13))):
            with self.assertRaises(ValueError):
                pool_hidden_states(invalid)

    def test_runtime_keeps_mixture_and_aligns_each_stem_once_per_track(self):
        runtime = object.__new__(_MERTRuntime)
        runtime.config, runtime.torch = self.config, torch
        runtime.reference_path, runtime.reference_waveform = None, None
        mixture = torch.linspace(-1, 1, 720001)
        guitar = self.root / "guitar.wav"
        runtime._load_audio = Mock(side_effect=lambda path: mixture if path == self.audio else mixture[:-1])
        runtime.processor = Mock(side_effect=lambda wave, **_: {"input_values": torch.from_numpy(wave).unsqueeze(0)})
        runtime.model = Mock(return_value=SimpleNamespace(hidden_states=tuple(torch.ones(1, 1, 768) for _ in range(13))))
        runtime.set_reference(self.audio)
        mix_result = list(runtime(self.audio))
        runtime.set_reference(self.audio)
        guitar_result = list(runtime(guitar))
        self.assertEqual(runtime._load_audio.call_count, 2)
        self.assertEqual([(a, b) for a, b, _ in mix_result], [(0, 720000), (720000, 720001)])
        self.assertEqual([(a, b) for a, b, _ in guitar_result], [(0, 720000), (720000, 720001)])
        self.assertEqual(runtime.last_audio_metadata["padded_end_samples"], 1)
        torch.testing.assert_close(runtime.reference_waveform, mixture, rtol=0, atol=0)
        self.assertEqual([len(call.args[0]) for call in runtime.processor.call_args_list], [720000, 400, 720000, 400])
        runtime._load_audio = Mock(return_value=torch.ones(710000))
        with self.assertRaisesRegex(ValueError, "differs"):
            list(runtime(guitar))

    def test_audio_loading_matches_author_mono_and_resampler(self):
        runtime = object.__new__(_MERTRuntime)
        runtime.config, runtime.torch = self.config, torch
        resample = Mock(return_value=torch.ones(400))
        runtime.torchaudio = SimpleNamespace(load=Mock(return_value=(torch.stack((torch.ones(200), torch.zeros(200))), 12000)),
                                             transforms=SimpleNamespace(Resample=Mock(return_value=resample)))
        self.assertEqual(runtime._load_audio(self.audio).numel(), 400)
        runtime.torchaudio.transforms.Resample.assert_called_once_with(12000, 24000)
        torch.testing.assert_close(resample.call_args.args[0], torch.full((200,), 0.5))

    def test_saved_format_true_tail_bounds_and_verified_resume(self):
        factory = Mock(return_value=self.extractor)
        report = extract_mert_segments(self.config, ("mixture", "guitar"), extractor_factory=factory)
        self.assertEqual((report["extracted"], report["failed"], report["minimum_padded_segments"]), (2, 0, 2))
        factory.assert_called_once_with(self.config, self.model_root)
        feature = np.load(self.folder / "segment_1.npy", allow_pickle=False)
        self.assertEqual((feature.shape, feature.dtype), ((1, 12, 768), np.dtype("float32")))
        marker = json.loads((self.folder / "segments.json").read_text())
        tail = marker["segments"][-1]
        self.assertEqual((tail["audio_samples"], tail["encoder_input_samples"], tail["minimum_padding_samples"]), (1, 400, 399))
        self.assertEqual(marker["settings"]["saved_hidden_states"], list(range(1, 13)))
        no_model = Mock(side_effect=AssertionError("Do not load encoder for reusable caches"))
        report = extract_mert_segments(self.config, ("mixture", "guitar"), extractor_factory=no_model)
        self.assertEqual((report["reused"], report["exit_code"]), (2, 0))
        no_model.assert_not_called()
        (self.folder / "segment_1.npy").write_bytes(b"corrupt")
        self.assertEqual(self.run_fixture()["failed"], 1)

    def test_changed_policy_model_and_reference_invalidate_cache(self):
        self.run_fixture()
        self.assertEqual(self.run_fixture(replace(self.config, short_segment_policy="error"))["failed"], 1)
        (self.model_root / "config.json").write_bytes(b"changed model identity")
        self.assertEqual(self.run_fixture()["failed"], 1)
        self.run_fixture(replace(self.config, overwrite=True))
        self.audio.write_bytes(b"changed reference identity")
        self.assertEqual(self.run_fixture()["failed"], 1)

    def test_failed_overwrite_preserves_previous_cache(self):
        self.run_fixture()
        before = {p.name: p.read_bytes() for p in self.folder.iterdir()}

        def broken(path):
            yield 0, 720000, np.zeros((1, 12, 768))
            raise RuntimeError("synthetic model failure")

        self.assertEqual(self.run_fixture(replace(self.config, overwrite=True), extractor=broken)["failed"], 1)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.folder.iterdir()})

    def test_failed_publication_restores_or_preserves_backup(self):
        self.run_fixture()
        before = {p.name: p.read_bytes() for p in self.folder.iterdir()}
        original_rename = Path.rename

        def fail_publish(path, target):
            if path.name.startswith(".1.") and ".backup." not in path.name and Path(target) == self.folder:
                raise OSError("synthetic publish failure")
            return original_rename(path, target)

        with patch.object(Path, "rename", fail_publish):
            self.assertEqual(self.run_fixture(replace(self.config, overwrite=True))["failed"], 1)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.folder.iterdir()})

        def fail_publish_and_restore(path, target):
            if path.name.startswith(".1.") and Path(target) == self.folder:
                raise OSError("synthetic publish/restore failure")
            return original_rename(path, target)

        with patch.object(Path, "rename", fail_publish_and_restore):
            self.assertEqual(self.run_fixture(replace(self.config, overwrite=True))["failed"], 1)
        backups = list(self.folder.parent.glob(".1.backup.*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(before, {p.name: p.read_bytes() for p in backups[0].iterdir()})

    def test_invalid_embeddings_and_legacy_folders_are_not_reused(self):
        self.folder.mkdir(parents=True)
        np.save(self.folder / "segment_0.npy", np.ones((1, 12, 768)))
        self.assertEqual(self.run_fixture()["failed"], 1)

        def invalid(path):
            yield 0, 720000, np.full((1, 12, 768), np.nan)

        self.assertEqual(self.run_fixture(replace(self.config, overwrite=True), extractor=invalid)["failed"], 1)
        self.assertFalse((self.folder / "run_manifest.json").exists())

    def test_counts_failures_and_fail_fast_are_reported(self):
        guitar = self.root / "guitar.wav"
        guitar.write_bytes(b"synthetic guitar identity")
        self.row["paths"]["guitar"] = str(guitar)
        self.manifest.write_text(json.dumps(self.row) + "\n")

        def varying(path):
            yield 0, 720000, np.ones((1, 12, 768))
            if path == self.audio:
                yield 720000, 720001, np.ones((1, 12, 768))

        report = self.run_fixture(sources=("mixture", "guitar"), extractor=varying)
        self.assertEqual(report["exit_code"], 1)
        self.assertEqual(report["segment_count_mismatches"][0]["counts"], {"mixture": 2, "guitar": 1})
        self.row["paths"].pop("mixture")
        self.manifest.write_text(json.dumps(self.row) + "\n")
        report = self.run_fixture(replace(self.config, continue_on_error=False), sources=("mixture", "guitar"))
        self.assertEqual((report["failed"], report["unprocessed"], report["exit_code"]), (1, 1, 1))

    def test_dry_run_has_no_audio_model_or_output_access(self):
        self.audio.unlink()
        no_model = Mock(side_effect=AssertionError("Dry run must not resolve model"))
        with patch("stemfusion.features.mert_segments.resolve_model", no_model):
            report = self.run_fixture(replace(self.config, dry_run=True))
        self.assertEqual((report["planned"], report["dry_run"]), (1, True))
        no_model.assert_not_called()
        self.assertFalse(self.config.output_dir.exists())

    def test_strict_yaml_profile_and_direct_cli_from_another_directory(self):
        preset = default_config_path("experiments/mtg_jamendo/mert.yaml")
        settings, provenance = load_configuration(preset, "mert_segment_extraction")
        self.assertEqual(settings["minimum_input_samples"], 400)
        self.assertEqual(settings["manifest"].name, "mtg_jamendo_features.jsonl")
        self.assertEqual(settings["source"], ["mixture", "guitar", "drums", "piano", "bass", "vocals"])
        self.assertEqual(len(provenance["config_files"]), 2)
        invalid = self.root / "invalid.yaml"
        invalid.write_text(preset.read_text().replace("../../features/mert_v1_95m.yaml",
            str(default_config_path("features/mert_v1_95m.yaml"))) + "unknown: true\n")
        with self.assertRaises(ConfigurationError):
            load_configuration(invalid, "mert_segment_extraction")
        script = Path(__file__).resolve().parents[2] / "src/stemfusion/experiments/mtg_jamendo/extract_mert.py"
        arguments = ["--manifest", str(self.manifest), "--output-dir", str(self.config.output_dir),
                     "--source", "mixture", "--split", "train", "--dry-run"]
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        result = subprocess.run([sys.executable, str(script), *arguments], cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("planned=1", result.stderr)
        env["PYTHONPATH"] = str(script.parents[4])
        result = subprocess.run([sys.executable, "-m", "stemfusion", "extract", "mtg-jamendo", "mert", *arguments],
                                cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("planned=1", result.stderr)
        self.assertFalse(self.config.output_dir.exists())


if __name__ == "__main__":
    unittest.main()

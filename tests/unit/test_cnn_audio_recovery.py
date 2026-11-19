"""Generic decoder recovery without changing CNN preprocessing or cohorts."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import librosa
import numpy as np
import soundfile as sf

from stemfusion.experiments import _cnn_extraction as cnn
from stemfusion.features import _cnn_audio as audio
from stemfusion.features.musicnn import save_cache


class CNNAudioRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "input.wav"
        sf.write(self.path, np.linspace(-.2, .2, 16000, dtype=np.float32), 16000, subtype="FLOAT")

    def test_healthy_primary_and_legitimate_short_audio_remain_unchanged(self):
        for samples in (16000, 128):
            with self.subTest(samples=samples):
                sf.write(self.path, np.full(samples, .1, np.float32), 16000, subtype="FLOAT")
                expected, _ = librosa.load(self.path, sr=16000, mono=True)
                with patch.object(audio, "_decode_ffmpeg") as fallback:
                    actual, metadata = audio.load_audio(self.path, .25)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(metadata["decoder"], "librosa")
                fallback.assert_not_called()

    def test_partial_primary_triggers_guarded_fallback(self):
        recovered = np.ones(16000, dtype=np.float32)
        with patch("librosa.load", return_value=(np.zeros(256, np.float32), 16000)), \
             patch.object(audio, "_decode_ffmpeg", return_value=(recovered, {"decoder": "ffmpeg"})) as fallback:
            actual, metadata = audio.load_audio(self.path, .25)
        self.assertIs(actual, recovered)
        fallback.assert_called_once_with(self.path, .25, None)
        self.assertIn("disagrees with its header", metadata["primary_error"])
        self.assertEqual(metadata["policy"], audio.POLICY)

    def test_nonfinite_primary_and_unavailable_recovery_are_errors(self):
        with patch("librosa.load", return_value=(np.full(16000, np.nan), 16000)), \
             patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(ValueError, "requires the system ffmpeg and ffprobe"):
                audio.load_audio(self.path, .25)

    def test_true_source_mismatch_remains_an_error(self):
        with patch.object(audio, "_decode_ffmpeg", side_effect=ValueError("still mismatched")) as fallback:
            with self.assertRaisesRegex(ValueError, "Source duration differs from mixture.*still mismatched"):
                audio.load_audio(self.path, .25, reference_samples=32000)
        fallback.assert_called_once_with(self.path, .25, 32000)

    def ffmpeg_results(self, *, duration=1.0, errors=b""):
        native = np.column_stack((np.full(44100, .6, np.float32), np.full(44100, -.2, np.float32)))
        probe = SimpleNamespace(stdout=json.dumps({"streams": [{
            "sample_rate": "44100", "channels": 2, "duration": str(duration),
        }]}), stderr="")
        decoded = SimpleNamespace(stdout=native.astype("<f4").tobytes(), stderr=errors)
        return [probe, decoded], native

    def test_ffmpeg_keeps_native_channels_then_uses_librosa_mono_and_resampling(self):
        results, native = self.ffmpeg_results()
        with patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"), \
             patch("subprocess.run", side_effect=results) as run, \
             patch.object(audio, "_version", return_value="fixture-version"):
            actual, metadata = audio._decode_ffmpeg(self.path, .25, 16000)
        expected = librosa.resample(librosa.to_mono(native.T), orig_sr=44100, target_sr=16000)
        np.testing.assert_array_equal(actual, expected)
        command = run.call_args_list[1].args[0]
        self.assertNotIn("-ar", command)
        self.assertNotIn("-ac", command)
        self.assertEqual(metadata["native_sample_rate"], 44100)
        self.assertEqual(metadata["native_channels"], 2)

    def test_ffmpeg_errors_and_duration_disagreement_are_not_silently_accepted(self):
        for duration, errors, message in (
            (1.0, b"Header missing", "reported damaged input"),
            (2.0, b"", "disagrees with its stream"),
        ):
            with self.subTest(duration=duration, errors=errors):
                results, _ = self.ffmpeg_results(duration=duration, errors=errors)
                with patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"), \
                     patch("subprocess.run", side_effect=results):
                    with self.assertRaisesRegex(ValueError, message):
                        audio._decode_ffmpeg(self.path, .25, None)

    def test_reference_revalidation_and_backups_preserve_strict_cache_identity(self):
        row = {"track_id": "fixture", "split": "train"}
        expected = {"source": "mixture", "checkpoint": "unchanged"}
        for source in cnn.SOURCES:
            path = self.root / "train" / source / "fixture.npz"
            save_cache(path, np.zeros(200, np.float32), {**expected, "aligned_samples": 128})
        path = self.root / "train/mixture/fixture.npz"
        self.assertTrue(cnn._reference_needs_validation(row, self.root, False, .25))
        self.assertFalse(cnn.cache_valid(path, expected, reference_samples=16000))
        with self.assertRaisesRegex(ValueError, "Incompatible existing cache"):
            cnn.cache_valid(path, {**expected, "checkpoint": "different"}, reference_samples=16000)
        before = path.read_bytes()
        backup = cnn._backup_repaired_cache(path)
        self.assertEqual(Path(backup["backup"]).read_bytes(), before)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(list(path.parent.glob("*.npz"))), 1)


if __name__ == "__main__":
    unittest.main()

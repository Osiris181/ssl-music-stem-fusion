from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from stemfusion.features._chord_cqt import cqt_features_with_tail_padding


class CQTRetryTests(unittest.TestCase):
    def setUp(self):
        self.parameters = SimpleNamespace(mp3={"song_hz": 22050, "inst_len": 10.0},
            feature={"n_bins": 144, "bins_per_octave": 24, "hop_length": 2048}, model={"timestep": 108})

    def test_only_one_sample_tail_is_padded_and_true_boundaries_are_recorded(self):
        waveform = np.arange(220501, dtype=np.float32)
        backend = SimpleNamespace(load=Mock(return_value=(waveform, 22050)),
                                  cqt=Mock(return_value=np.ones((144, 1), dtype=np.complex64)))
        with patch.dict("sys.modules", {"librosa": backend}):
            feature, timing, duration, metadata = cqt_features_with_tail_padding("fixture.wav", self.parameters)
        self.assertEqual(feature.shape, (144, 2))
        self.assertEqual(timing, 10 / 108)
        self.assertEqual(duration, len(waveform) / 22050)
        calls = backend.cqt.call_args_list
        np.testing.assert_array_equal(calls[0].args[0], waveform[:220500])
        np.testing.assert_array_equal(calls[1].args[0], [waveform[-1], 0])
        self.assertEqual(metadata["padded_samples"], 1)
        self.assertEqual(metadata["padded_blocks"], [{"start_sample": 220500, "end_sample": 220501,
            "audio_samples": 1, "cqt_input_samples": 2, "right_zero_padding_samples": 1}])

    def test_valid_partial_blocks_are_not_padded_or_dropped(self):
        waveform = np.arange(220503, dtype=np.float32)
        backend = SimpleNamespace(load=Mock(return_value=(waveform, 22050)),
                                  cqt=Mock(return_value=np.ones((144, 1), dtype=np.complex64)))
        with patch.dict("sys.modules", {"librosa": backend}):
            *_, metadata = cqt_features_with_tail_padding("fixture.wav", self.parameters)
        np.testing.assert_array_equal(backend.cqt.call_args_list[1].args[0], waveform[220500:])
        self.assertEqual(metadata["padded_samples"], 0)
        self.assertFalse(metadata["padded_blocks"])

    def test_empty_invalid_audio_and_unvalidated_frontends_are_not_repaired(self):
        for waveform in (np.empty(0), np.array([np.nan]), np.ones((1, 2))):
            backend = SimpleNamespace(load=Mock(return_value=(waveform, 22050)))
            with patch.dict("sys.modules", {"librosa": backend}), self.assertRaises(ValueError):
                cqt_features_with_tail_padding("fixture.wav", self.parameters)
        self.parameters.mp3["song_hz"] = 24000
        with patch.dict("sys.modules", {"librosa": SimpleNamespace()}), self.assertRaisesRegex(ValueError, "pinned BTC"):
            cqt_features_with_tail_padding("fixture.wav", self.parameters)


if __name__ == "__main__":
    unittest.main()

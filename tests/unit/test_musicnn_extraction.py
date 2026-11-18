"""Pure array tests: no TensorFlow imports, checkpoint downloads or GPU use."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from stemfusion.features.musicnn import align_audio, iter_windows, pool_track, save_cache
from stemfusion.experiments.gtzan.extraction_cnn import cache_valid, read_records, prepare_runtime


class MusicNNExtractionTests(unittest.TestCase):
    def test_runtime_setup_restarts_only_when_needed(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "nvidia/example/lib"
            library.mkdir(parents=True)
            with patch("site.getsitepackages", return_value=[directory]), \
                 patch.dict("os.environ", {"PATH": "/usr/bin"}, clear=True), \
                 patch("os.execve") as restart:
                prepare_runtime()
                restart.assert_called_once()
                environment = restart.call_args.args[2]
                self.assertEqual(environment["TF_USE_LEGACY_KERAS"], "1")
                self.assertEqual(environment["LD_LIBRARY_PATH"], str(library))
                with patch.dict("os.environ", environment, clear=True):
                    restart.reset_mock()
                    prepare_runtime()
                    restart.assert_not_called()

    def test_tail_coverage(self):
        for frames, windows, padding in [(1, 1, 186), (186, 1, 1), (187, 1, 0),
                                         (188, 2, 186), (374, 2, 0), (375, 3, 186)]:
            with self.subTest(frames=frames):
                mel = np.ones((frames, 96), dtype=np.float16)
                patches = np.concatenate(list(iter_windows(mel, 2)))
                self.assertEqual(patches.shape, (windows, 187, 96))
                flat = patches.reshape(-1, 96)
                np.testing.assert_array_equal(flat[:frames], mel)
                self.assertEqual(np.count_nonzero(flat[frames:]), 0)
                self.assertEqual(len(flat) - frames, padding)

    def test_alignment(self):
        np.testing.assert_array_equal(align_audio(np.ones(99), 100, .25)[-2:], [1, 0])
        self.assertEqual(len(align_audio(np.ones(101), 100, .25)), 100)
        with self.assertRaises(ValueError):
            align_audio(np.ones(5000), 100, .25)
        with self.assertRaises(ValueError):
            align_audio(np.array([np.nan]), 1, .25)

    def test_pooling_weights_windows_not_batches(self):
        class Model:
            def predict(self, batch):
                return np.repeat(batch[:, 0, :1], 200, axis=1)
        mel = np.concatenate([np.full((187, 96), x) for x in (1., 2., 9.)])
        a, count = pool_track(Model(), mel, 2)
        b, _ = pool_track(Model(), mel, 1)
        self.assertEqual(count, 3)
        np.testing.assert_array_equal(a, np.full(200, 4., dtype=np.float32))
        np.testing.assert_array_equal(a, b)

    def test_cache_resume_and_incompatible_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vector.npz"
            expected = {"checkpoint": "MSD_musicnn", "source": "mixture"}
            self.assertFalse(cache_valid(path, expected))
            save_cache(path, np.zeros(200, dtype=np.float32), expected)
            self.assertTrue(cache_valid(path, expected))
            with self.assertRaises(ValueError):
                cache_valid(path, {**expected, "source": "guitar"})

    def test_repeated_mixture_shapes(self):
        mixture = np.arange(200, dtype=np.float32)
        for count in (1, 4, 5, 6):
            vector = np.concatenate([mixture] * count)
            self.assertEqual(vector.shape, (200 * count,))
            for piece in np.split(vector, count):
                np.testing.assert_array_equal(piece, mixture)

    def test_manifest_rejects_cross_split_duplicate(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            record = {"track_id": "blues.00000", "label": 0, "split": "train",
                      "paths": {source: "unused" for source in
                                ("mixture", "guitar", "drums", "piano", "bass", "vocals")}}
            path.write_text(json.dumps(record) + "\n" + json.dumps({**record, "split": "test"}))
            with self.assertRaises(ValueError):
                read_records(path)


if __name__ == "__main__":
    unittest.main()

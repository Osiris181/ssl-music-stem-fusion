"""Genre, mood/theme and emotion adapters for shared CNN extraction."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import yaml

from stemfusion.experiments import _cnn_extraction as cnn
from stemfusion.features.musicnn import SOURCES, save_cache


class CNNDatasetExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        audio = self.root / "audio.mp3"
        audio.write_bytes(b"fixture audio; decoder/model are injected")
        self.paths = {source: str(audio) for source in (*SOURCES, "other")}

    def write_manifest(self, dataset):
        if dataset == "fma_medium":
            rows = [
                {"track_id": f"{index * 16 + label:06d}", "genre": f"genre_{label}",
                 "label": label, "split": split, "paths": self.paths}
                for index, split in enumerate(("train", "validation", "test")) for label in range(16)
            ]
        elif dataset in ("pmemo", "deam", "emomusic"):
            rows = [
                {"dataset": dataset, "track_id": str(index), "relative_path": f"{index}.mp3",
                 "split": split, "paths": self.paths, "duration_seconds": 3.0,
                 "annotations": {"valence": .575, "arousal": .4} if dataset == "pmemo"
                                else {"valence": 5.75, "arousal": 4.0}}
                for index, split in enumerate(("train", "validation", "test"), 1)
            ]
        else:
            rows = [
                {"track_id": f"track_{index:07d}", "relative_path": f"{index:02d}/{index}.mp3",
                 "split": split, "paths": self.paths, "duration_seconds": 30.0,
                 "tags": [f"mood/theme---tag_{tag}" for tag in range(56)]}
                for index, split in enumerate(("train", "validation", "test"), 1)
            ]
        manifest = self.root / f"{dataset}.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return manifest, rows

    def configuration(self, dataset, manifest):
        path = self.root / f"{dataset}.yaml"
        path.write_text(yaml.safe_dump({
            "dataset": dataset,
            "paths": {"manifest": str(manifest), "output_dir": str(self.root / dataset),
                      "report": str(self.root / f"{dataset}_report.json")},
            "runtime": {"device": "/CPU:0", "batch_size": 16, "gpu_memory_mb": 768},
            "audio": {"maximum_source_duration_difference_seconds": .25},
        }))
        return path

    def test_fma_ids_classes_and_metadata_plan(self):
        path, rows = self.write_manifest("fma_medium")
        records = cnn.read_records(path, dataset="fma_medium")
        self.assertEqual(records, rows)
        self.assertEqual(records[0]["track_id"], "000000")
        self.assertEqual(max(row["label"] for row in records), 15)
        with patch.object(cnn, "OfficialMusicNN") as model, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cnn.main(["--config", str(self.configuration("fma_medium", path)), "--dry-run"],
                                      dataset="fma_medium"), 0)
            model.assert_not_called()

    def test_jamendo_preserves_tags_and_validates_vocabulary(self):
        path, rows = self.write_manifest("mtg_jamendo")
        self.assertEqual(cnn.read_records(path, dataset="mtg_jamendo"), rows)
        for row in rows:
            row["tags"] = row["tags"][:-1]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "56-tag"):
            cnn.read_records(path, dataset="mtg_jamendo")

    def test_all_target_formats_extract_and_resume(self):
        for dataset in ("fma_medium", "mtg_jamendo", "pmemo", "deam", "emomusic"):
            with self.subTest(dataset=dataset):
                path, rows = self.write_manifest(dataset)
                config = self.configuration(dataset, path)
                args = ["--config", str(config), "--limit-per-split", "1", "--gpu-memory-mb", "4096"]
                with patch.object(cnn, "OfficialMusicNN") as factory, \
                     patch("soundfile.info", return_value=SimpleNamespace(duration=3.0)), \
                     patch("librosa.load", return_value=(np.zeros(48000, dtype=np.float32), 16000)) as load, \
                     patch.object(cnn, "log_mel", return_value=np.zeros((188, 96), dtype=np.float16)), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    model = factory.return_value
                    model.provenance = {"checkpoint": "MSD_musicnn"}
                    model.predict.side_effect = lambda batch: np.ones((len(batch), 200), dtype=np.float32)
                    self.assertEqual(cnn.main(args, dataset=dataset), 0)
                    factory.assert_called_once_with(device="/CPU:0", gpu_memory_mb=4096)
                    files = list((self.root / dataset).glob("*/*/*.npz"))
                    self.assertEqual(len(files), 18)
                    for file in files:
                        with np.load(file, allow_pickle=False) as cache:
                            metadata = json.loads(str(cache["metadata"]))
                            self.assertEqual(cache["embedding"].shape, (200,))
                            self.assertEqual(metadata["windows"], 2)
                            self.assertEqual(metadata["padded_frames"], 186)
                            self.assertEqual(metadata["runtime"]["gpu_memory_mb"], 4096)
                            if dataset == "mtg_jamendo":
                                self.assertEqual(metadata["tags"], rows[0]["tags"])
                                self.assertIn("relative_path", metadata)
                                self.assertNotIn("label", metadata)
                            elif dataset in ("pmemo", "deam", "emomusic"):
                                self.assertEqual(metadata["annotations"], rows[0]["annotations"])
                                self.assertEqual(metadata["relative_path"], f"{metadata['track_id']}.mp3")
                                self.assertNotIn("label", metadata)
                                self.assertNotIn("tags", metadata)
                            else:
                                self.assertIn("label", metadata)
                    times = {file: file.stat().st_mtime_ns for file in files}
                    load.reset_mock()
                    model.predict.reset_mock()
                    self.assertEqual(cnn.main(args, dataset=dataset), 0)
                    load.assert_not_called()
                    model.predict.assert_not_called()
                    self.assertEqual(times, {file: file.stat().st_mtime_ns for file in files})
                report = json.loads((self.root / f"{dataset}_report.json").read_text())
                self.assertEqual(report["reused"], 18)
                self.assertEqual(report["failed"], [])
                if dataset in ("pmemo", "deam", "emomusic"):
                    self.assertEqual(report["targets"], ["valence", "arousal"])
                    self.assertEqual(report["annotation_scale"],
                                     "original_0_1" if dataset == "pmemo" else "original_1_9")
                    self.assertFalse(report["binary_labels_created"])

    def test_partial_cohort_revalidates_reference_and_repairs_only_wrong_length_cache(self):
        path, rows = self.write_manifest("fma_medium")
        config = self.configuration("fma_medium", path)
        args = ["--config", str(config), "--limit-per-split", "1"]
        with patch.object(cnn, "OfficialMusicNN") as factory, \
             patch("soundfile.info", return_value=SimpleNamespace(duration=3.0)), \
             patch("librosa.load", return_value=(np.zeros(48000, dtype=np.float32), 16000)), \
             patch.object(cnn, "log_mel", return_value=np.zeros((188, 96), dtype=np.float16)), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model = factory.return_value
            model.provenance = {"checkpoint": "MSD_musicnn"}
            model.predict.side_effect = lambda batch: np.ones((len(batch), 200), dtype=np.float32)
            self.assertEqual(cnn.main(args, dataset="fma_medium"), 0)
            mixture_path = self.root / "fma_medium/train/mixture/000000.npz"
            with np.load(mixture_path, allow_pickle=False) as cache:
                metadata = json.loads(str(cache["metadata"]))
            metadata.update(original_samples=256, aligned_samples=256,
                            spectrogram_frames=2, windows=1, padded_frames=185)
            save_cache(mixture_path, np.zeros(200, np.float32), metadata)
            original = mixture_path.read_bytes()
            for source in SOURCES[1:]:
                (self.root / f"fma_medium/train/{source}/000000.npz").unlink()
            valid_paths = [file for file in (self.root / "fma_medium").glob("*/*/*.npz") if file != mixture_path]
            before = {file: (file.read_bytes(), file.stat().st_mtime_ns) for file in valid_paths}
            model.predict.reset_mock()
            self.assertEqual(cnn.main(args, dataset="fma_medium"), 0)
            self.assertEqual(model.predict.call_count, 6)
            self.assertEqual(before, {file: (file.read_bytes(), file.stat().st_mtime_ns) for file in before})
            with np.load(mixture_path, allow_pickle=False) as cache:
                repaired = json.loads(str(cache["metadata"]))
                np.testing.assert_array_equal(cache["embedding"], np.ones(200, np.float32))
            self.assertEqual(repaired["aligned_samples"], 48000)
            self.assertEqual(Path(repaired["audio_repair"]["backup"]).read_bytes(), original)
            report = json.loads((self.root / "fma_medium_report.json").read_text())
            self.assertEqual((report["extracted"], report["reused"], len(report["repaired"])), (6, 12, 1))
            self.assertEqual(report["failed"], [])

    def test_unrecoverable_source_remains_failed_and_preserves_completed_views(self):
        for source in SOURCES:
            source_path = self.root / f"{source}.mp3"
            source_path.write_bytes(b"injected decoder fixture")
            self.paths[source] = str(source_path)
        path, _ = self.write_manifest("fma_medium")
        config = self.configuration("fma_medium", path)
        args = ["--config", str(config), "--limit-per-split", "1", "--split", "train"]

        def decode(source_path, tolerance, *, reference_samples=None):
            self.assertEqual(tolerance, .25)
            if source_path.name == "drums.mp3":
                raise ValueError("FFmpeg reported damaged input; regenerate the audio")
            return np.zeros(48000, np.float32), {"policy": "fixture", "decoder": "librosa"}

        with patch.object(cnn, "OfficialMusicNN") as factory, \
             patch.object(cnn._cnn_audio, "load_audio", side_effect=decode), \
             patch.object(cnn, "log_mel", return_value=np.zeros((188, 96), dtype=np.float16)), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model = factory.return_value
            model.provenance = {"checkpoint": "MSD_musicnn"}
            model.predict.side_effect = lambda batch: np.ones((len(batch), 200), np.float32)
            self.assertEqual(cnn.main(args, dataset="fma_medium"), 1)
            files = list((self.root / "fma_medium").glob("*/*/*.npz"))
            self.assertEqual(len(files), 5)
            before = {file: (file.read_bytes(), file.stat().st_mtime_ns) for file in files}
            model.predict.reset_mock()
            self.assertEqual(cnn.main(args, dataset="fma_medium"), 1)
            model.predict.assert_not_called()
            self.assertEqual(before, {file: (file.read_bytes(), file.stat().st_mtime_ns) for file in files})
            report = json.loads((self.root / "fma_medium_report.json").read_text())
            self.assertEqual((report["extracted"], report["reused"], len(report["failed"])), (0, 5, 1))
            self.assertEqual(report["failed"][0]["source"], "drums")


if __name__ == "__main__":
    unittest.main()

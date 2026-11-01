from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from stemfusion.cli import build_parser
from stemfusion.data.ismir04 import LABEL_MAPPING, SOURCES
from stemfusion.experiments._extraction import musicfm_config_from_args
from stemfusion.features._musicfm import (
    MusicFMExtractionConfig, MusicFMExtractionError, _MusicFMRuntime,
    _chunk_ranges, extract_musicfm_sources,
)


class ISMIR04ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model_root = self.root / "musicfm"
        (self.model_root / "data").mkdir(parents=True)
        (self.model_root / "data/pretrained_msd.pt").write_bytes(b"synthetic")
        (self.model_root / "data/msd_stats.json").write_text("{}")
        self.manifest = self.root / "manifest.jsonl"
        self.records = []
        for split in ("train", "validation", "test"):
            for genre, label in LABEL_MAPPING.items():
                track_id = f"{split}__{genre}__0123456789abcdef"
                paths = {}
                for source in SOURCES:
                    path = self.root / "audio" / track_id / f"{source}.wav"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"synthetic audio")
                    paths[source] = str(path)
                self.records.append({"track_id": track_id, "split": split, "genre": genre, "label": label, "paths": paths})
        self.manifest.write_text("".join(json.dumps(record) + "\n" for record in self.records))
        self.config = MusicFMExtractionConfig(self.manifest, self.root / "features", "mixture", self.model_root,
            device="cpu", dataset="ismir04", expected_dimension=3, sample_rate=1, chunk_seconds=30)

    def tearDown(self):
        self.temporary.cleanup()

    def runtime(self, waveform):
        runtime = _MusicFMRuntime.__new__(_MusicFMRuntime)
        runtime.config = self.config
        runtime.torch = torch
        runtime._load_audio = lambda _: waveform
        runtime.model = self.Model()
        return runtime

    class Model:
        def __init__(self):
            self.calls = []
            self.embeddings = []

        def get_latent(self, audio, layer_ix):
            self.calls.append((audio.clone(), layer_ix))
            frames = audio[:, ::2]
            embedding = torch.stack((frames, frames.square(), frames + 1), dim=-1)
            self.embeddings.append(embedding)
            return embedding

    def test_balanced_chunks_cover_full_audio_without_tiny_tail(self):
        ranges = _chunk_ranges(61, 1, 30)
        self.assertEqual(ranges, [(0, 20), (20, 40), (40, 61)])
        self.assertEqual(sum(end - start for start, end in ranges), 61)
        self.assertTrue(all(end - start <= 30 for start, end in ranges))
        self.assertEqual(_chunk_ranges(30, 1, 30), [(0, 30)])
        for value in (0, float("nan"), float("inf")):
            with self.assertRaises(MusicFMExtractionError):
                _chunk_ranges(61, 1, value)

    def test_chunk_pooling_weights_actual_latent_frames_and_passes_layer_seven(self):
        waveform = torch.arange(61, dtype=torch.float32)
        runtime = self.runtime(waveform)
        result = runtime(Path("unused.wav"))
        self.assertEqual([layer for _, layer in runtime.model.calls], [7, 7, 7])
        self.assertTrue(torch.equal(torch.cat([audio.squeeze(0) for audio, _ in runtime.model.calls]), waveform))
        expected = torch.cat(runtime.model.embeddings, dim=1).mean(dim=1).squeeze(0)
        self.assertTrue(torch.allclose(result, expected))
        equal_chunk_mean = torch.stack([embedding.mean(dim=1).squeeze(0) for embedding in runtime.model.embeddings]).mean(dim=0)
        self.assertFalse(torch.allclose(result, equal_chunk_mean))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device.type, "cpu")

    def test_single_pass_preserves_existing_mean_pooling(self):
        runtime = self.runtime(torch.arange(61, dtype=torch.float32))
        runtime.config = replace(self.config, chunk_seconds=None)
        result = runtime(Path("unused.wav"))
        self.assertEqual(len(runtime.model.calls), 1)
        self.assertTrue(torch.equal(result, runtime.model.embeddings[0].mean(dim=1).squeeze(0)))

    def test_dataset_yaml_and_cli_choose_explicit_chunk_policy(self):
        parser = build_parser()
        for dataset, chunks in (("fma-medium", None), ("gtzan", None), ("ismir04", 30)):
            args = parser.parse_args(["extract", dataset])
            config = musicfm_config_from_args(args, source="mixture")
            self.assertEqual(config.layer_index, 7)
            self.assertIsNone(config.max_seconds)
            self.assertEqual(config.chunk_seconds, chunks)
        args = parser.parse_args(["extract", "ismir04", "--no-chunking"])
        self.assertIsNone(musicfm_config_from_args(args, source="mixture").chunk_seconds)

    def test_shared_extractor_writes_unique_ismir04_keys_and_chunk_reports(self):
        configs = [replace(self.config, source=source) for source in SOURCES]
        summaries = extract_musicfm_sources(configs, feature_extractor=lambda _: torch.ones(3))
        self.assertEqual(sum(summary.succeeded for summary in summaries), 108)
        path = self.config.output_dir / "ismir04_musicfm_mix_test.pkl"
        with path.open("rb") as stream:
            cache = pickle.load(stream)
        self.assertEqual(set(cache), {record["track_id"] for record in self.records if record["split"] == "test"})
        report = json.loads((self.config.output_dir / "extraction_report_musicfm_mixture.json").read_text())
        self.assertEqual(report["chunk_seconds"], 30)
        self.assertEqual(report["chunk_pooling"], "latent_frame_mean")
        self.assertEqual(report["chunk_partition"], "balanced_non_overlapping")
        self.assertEqual(report["layer_index"], 7)
        self.assertEqual(report["resolved_config"]["dataset"], "ismir04")

    def test_direct_entry_dry_run_does_not_write_features(self):
        script = Path(__file__).resolve().parents[2] / "src/stemfusion/experiments/ismir04/extraction.py"
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        result = subprocess.run([sys.executable, str(script), "--manifest", str(self.manifest),
            "--musicfm-root", str(self.model_root), "--output-dir", str(self.config.output_dir),
            "--dry-run"], cwd=self.root, env=env, capture_output=True, text=True, check=True)
        self.assertFalse(self.config.output_dir.exists())
        self.assertIn("Source 6/6", result.stderr)


if __name__ == "__main__":
    unittest.main()

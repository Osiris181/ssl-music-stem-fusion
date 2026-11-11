"""EmoMusic fixed metadata and unchanged MERT/BTC extraction behavior."""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from stemfusion.cli import build_parser
from stemfusion.configuration import default_config_path, load_configuration
from stemfusion.data.emomusic import build_feature_manifest, read_feature_manifest
from stemfusion.features import chords_keys, emotion_dataset, emotion_extraction, mert_segments


class ParameterError(ValueError):
    pass


class EmoMusicExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.splits = {}
        for split, track_id in (("train", "2"), ("validation", "3"), ("test", "4")):
            self.splits[split] = self.root / f"{split}.txt"
            self.splits[split].write_text(track_id + "\n")
        self.annotations = self.root / "static_annotations.csv"
        self.annotations.write_text("song_id,valence_mean,valence_std,arousal_mean,arousal_std\n"
                                    "2,3.1,0.94,3.0,0.63\n3,3.5,1.75,3.3,1.62\n4,5.7,1.42,5.5,1.63\n")
        self.manifest = self.root / "features.jsonl"
        self.adapter = emotion_dataset.EmotionDatasetAdapter("emomusic", read_feature_manifest)

    def tearDown(self):
        self.temp.cleanup()

    def build(self, **kwargs):
        return build_feature_manifest(split_files=self.splits, annotations=self.annotations,
            mixture_root=self.root / "clips_45sec", htdemucs_root=self.root / "htdemucs_6s",
            output=self.manifest, **kwargs)

    def prepare_audio(self):
        self.build()
        for row in read_feature_manifest(self.manifest):
            for source in ("mixture", "guitar"):
                path = Path(row["paths"][source])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic audio identity")

    def test_preserves_supplied_scores_splits_and_audio_layout(self):
        report = self.build()
        rows = read_feature_manifest(self.manifest)
        self.assertEqual([row["track_id"] for row in rows], ["2", "3", "4"])
        self.assertEqual(rows[0]["annotations"], {"valence": 3.1, "arousal": 3.0})
        self.assertEqual(rows[0]["paths"]["mixture"], str(self.root / "clips_45sec/2.mp3"))
        self.assertEqual(rows[0]["paths"]["guitar"], str(self.root / "htdemucs_6s/2/guitar.mp3"))
        self.assertFalse(report["binary_labels_created"])
        self.assertEqual(report["annotation_aggregation"], "use_supplied_song_level_means")
        with self.assertRaisesRegex(ValueError, "exists"):
            self.build()

    def test_dry_manifest_never_checks_audio_or_writes(self):
        self.assertEqual(self.build(dry_run=True)["records"], 3)
        self.assertFalse(self.manifest.exists())
        self.assertFalse(self.manifest.with_suffix(".report.json").exists())

    def test_overlap_missing_and_unassigned_annotations_fail_explicitly(self):
        self.splits["test"].write_text("2\n")
        with self.assertRaisesRegex(ValueError, "Repeated"):
            self.build(dry_run=True)
        self.splits["test"].write_text("5\n")
        with self.assertRaisesRegex(ValueError, "lacks"):
            self.build(dry_run=True)
        self.splits["test"].write_text("4\n")
        self.annotations.write_text(self.annotations.read_text() + "5,5,1,5,1\n")
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            self.build(dry_run=True)

    def test_duplicate_annotations_and_invalid_scales_are_rejected(self):
        original = self.annotations.read_text()
        self.annotations.write_text(original + "2,5,1,5,1\n")
        with self.assertRaisesRegex(ValueError, "Repeated annotated"):
            self.build(dry_run=True)
        self.annotations.write_text(original.replace("3.1", "0.31"))
        with self.assertRaisesRegex(ValueError, "1-9"):
            self.build(dry_run=True)

    def test_numeric_ids_paths_and_dataset_binding_are_strict(self):
        self.build()
        row = read_feature_manifest(self.manifest)[0]
        for changes in ({"dataset": "pmemo"}, {"track_id": "02"}, {"relative_path": "../2.mp3"},
                        {"paths": {"mixture": "relative.mp3"}}):
            self.manifest.write_text(json.dumps({**row, **changes}) + "\n")
            with self.assertRaises(ValueError):
                self.adapter.read_manifest(self.manifest)
        with self.assertRaises(ValueError):
            emotion_dataset.EmotionDatasetAdapter("../emomusic", read_feature_manifest)

    def test_scientific_defaults_match_both_existing_datasets(self):
        for method, role in (("mert", "mert_segment_extraction"), ("chords_keys", "chord_key_extraction")):
            values, _ = load_configuration(default_config_path(f"experiments/emomusic/{method}.yaml"), role)
            for dataset in ("pmemo", "deam", "mtg_jamendo"):
                reference, _ = load_configuration(default_config_path(f"experiments/{dataset}/{method}.yaml"), role)
                paths = {"manifest", "output_dir", "report"}
                self.assertEqual({key: value for key, value in values.items() if key not in paths},
                                 {key: value for key, value in reference.items() if key not in paths})

    def test_scheduler_matches_frozen_pmemo_except_adapter_binding(self):
        for name in ("extract_mert_segments", "extract_chords_keys"):
            source = inspect.getsource(getattr(emotion_dataset, name))
            source = source.replace("adapter: EmotionDatasetAdapter, ", "")
            source = source.replace("adapter.read_manifest(", "read_feature_manifest(")
            source = source.replace('"dataset": adapter.dataset', '"dataset": "pmemo"')
            source = source.replace("adapter.identity()", "_adapter_identity()")
            reference = inspect.getsource(getattr(emotion_extraction, name))
            self.assertEqual(ast.dump(ast.parse(source)), ast.dump(ast.parse(reference)))

    def test_direct_entrypoints_work_from_another_directory(self):
        directory = Path(__file__).resolve().parents[2] / "src/stemfusion/experiments/emomusic"
        for name in ("prepare_feature_manifest.py", "extract_mert.py", "extract_chords_keys.py"):
            result = subprocess.run([sys.executable, str(directory / name), "--help"], cwd=self.root,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_yaml_defaults_and_explicit_selection(self):
        parser = build_parser()
        for method in ("mert", "chords-keys"):
            args = parser.parse_args(["extract", "emomusic", method, "--source", "guitar", "--split", "test", "--dry-run"])
            self.assertEqual(args.source, ["guitar"])
            self.assertEqual(args.splits, ["test"])
            self.assertEqual(args.emotion_dataset, "emomusic")
            self.assertEqual(args.yaml_provenance["yaml_settings_before_cli_overrides"]["dataset"], "emomusic")
        args = parser.parse_args(["data", "emomusic-feature-manifest", "--dry-run"])
        self.assertEqual(args.annotations.name, "static_annotations.csv")

    def test_existing_deam_stem_root_is_preserved(self):
        values, _ = load_configuration(default_config_path("experiments/emomusic/feature_manifest.yaml"),
                                       "emomusic_feature_manifest")
        self.assertEqual(values["mixture_root"], Path("/mnt/HDD/emoMusic/clips_45sec"))
        self.assertEqual(values["htdemucs_root"], Path("/mnt/HDD/osiris181/DEAM_separated/htdemucs_6s"))

    def test_annotation_columns_are_not_swapped(self):
        self.build()
        row = read_feature_manifest(self.manifest)[0]
        self.assertEqual(row["annotations"], {"valence": 3.1, "arousal": 3.0})

    def test_dataset_switch_reuses_the_same_scheduler_and_config_classes(self):
        from stemfusion.experiments import _emotion_features
        parser = build_parser()
        for dataset in ("deam", "emomusic"):
            args = parser.parse_args(["extract", dataset, "mert", "--dry-run"])
            self.assertIs(args.command_handler, _emotion_features.run_from_args)

    def test_extraction_planning_reads_no_audio_and_loads_no_models(self):
        self.build()
        mert = mert_segments.MERTSegmentConfig(self.manifest, self.root / "mert", dry_run=True)
        harmony = chords_keys.ChordKeyExtractionConfig(self.manifest, self.root / "harmony", self.root,
            self.root / "btc.yaml", self.root / "btc.pt", self.root / "report.json", dry_run=True)
        factory = Mock(side_effect=AssertionError("Model must not load"))
        self.assertEqual(emotion_dataset.extract_mert_segments(mert, ("mixture",), adapter=self.adapter,
                         extractor_factory=factory)["planned"], 3)
        self.assertEqual(emotion_dataset.extract_chords_keys(harmony, adapter=self.adapter,
                         runtime_factory=factory)["planned"], 18)
        self.assertFalse(harmony.report.exists())
        self.assertFalse(mert.output_dir.exists())

    def test_shared_mert_writer_retains_tails_and_verifies_cache(self):
        self.prepare_audio()
        model = self.root / "model"
        model.mkdir()
        for name in mert_segments.MODEL_FILES:
            (model / name).write_bytes(b"synthetic model")
        config = mert_segments.MERTSegmentConfig(self.manifest, self.root / "mert", model_path=model,
                                                 device="cpu", splits=("train",))
        def extract(audio):
            yield 0, 720000, np.ones((1, 12, 768), dtype=np.float32)
            yield 720000, 720001, np.ones((1, 12, 768), dtype=np.float32)
        factory = Mock(return_value=extract)
        report = emotion_dataset.extract_mert_segments(config, ("mixture", "guitar"), adapter=self.adapter,
                                                       extractor_factory=factory)
        self.assertEqual((report["dataset"], report["extracted"], report["minimum_padded_segments"]), ("emomusic", 2, 2))
        tail = config.output_dir / "mert_30s/2/segment_1.npy"
        self.assertEqual(np.load(tail, allow_pickle=False).shape, (1, 12, 768))
        marker = json.loads((tail.parent / "segments.json").read_text())
        self.assertEqual(marker["segments"][-1]["minimum_padding_samples"], 399)
        factory.reset_mock()
        again = emotion_dataset.extract_mert_segments(config, ("mixture", "guitar"), adapter=self.adapter,
                                                      extractor_factory=factory)
        self.assertEqual(again["reused"], 2)
        factory.assert_not_called()
        tail.write_bytes(b"corrupt array")
        bad = emotion_dataset.extract_mert_segments(config, ("mixture",), adapter=self.adapter,
                                                    extractor_factory=factory)
        self.assertEqual((bad["failed"], bad["exit_code"]), (1, 1))

    def test_shared_harmony_writer_retries_only_supported_failures(self):
        self.prepare_audio()
        config = chords_keys.ChordKeyExtractionConfig(self.manifest, self.root / "harmony", self.root,
            self.root / "btc.yaml", self.root / "btc.pt", self.root / "report.json",
            sources=("mixture", "guitar"), splits=("train",), device="cpu")
        class Runtime:
            def __call__(self, audio, output):
                if audio.name == "guitar.mp3":
                    raise ParameterError("Input signal length=1 is too short for 6-octave CQT")
                return self.retry_with_cqt_padding(audio, output)

            def retry_with_cqt_padding(self, audio, output):
                for name in ("chords.lab", "normalized_chords.lab"):
                    (output / name).write_text("0.000 1.000 C\n1.000 2.000 G\n")
                (output / "chords.mid").write_bytes(b"synthetic MIDI")
                (output / "key.lab").write_text("C major\n")
                return {"chord_intervals": 2, "midi_notes": 5, "key": "C major", "key_status": "estimated", "key_error": None}
        factory = Mock(return_value=Runtime())
        with patch.object(chords_keys, "_settings", return_value={"fixture": True}):
            report = emotion_dataset.extract_chords_keys(config, adapter=self.adapter, runtime_factory=factory)
            self.assertEqual((report["repaired"], report["failed"], report["exit_code"]), (1, 0, 0))
            successful = config.output_dir / "mixture/2/metadata.json"
            before = successful.read_bytes()
            factory.reset_mock()
            again = emotion_dataset.extract_chords_keys(config, adapter=self.adapter, runtime_factory=factory)
            self.assertEqual(again["reused"], 2)
            factory.assert_not_called()
            self.assertEqual(successful.read_bytes(), before)

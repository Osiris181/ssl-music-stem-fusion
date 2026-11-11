"""Fixed PMEmo metadata, consistent extraction, safe resume and CQT retry."""

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
from stemfusion.data.pmemo import build_feature_manifest, read_feature_manifest
from stemfusion.features import chords_keys, emotion_extraction, mert_segments


class ParameterError(ValueError):
    pass


class PMEmoExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.splits = {}
        for name, track_id in (("train", "1"), ("validation", "4"), ("test", "5")):
            self.splits[name] = self.root / f"{name}.txt"
            self.splits[name].write_text(track_id + "\n")
        self.annotations = self.root / "annotations.csv"
        self.annotations.write_text("musicId,Arousal(mean),Valence(mean)\n1,0.4,0.575\n4,0.25,0.3\n5,0.1,0.2\n")
        self.metadata = self.root / "metadata.csv"
        self.metadata.write_text("musicId,fileName,duration\n1,1.mp3,30.1\n4,4.mp3,31\n5,5.mp3,32\n")
        self.manifest = self.root / "manifest.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def build(self, **kwargs):
        return build_feature_manifest(split_files=self.splits, annotations=self.annotations, metadata=self.metadata,
            mixture_root=self.root / "audio", htdemucs_root=self.root / "stems", output=self.manifest, **kwargs)

    def prepare_audio(self):
        self.build()
        rows = read_feature_manifest(self.manifest)
        for row in rows:
            for name in ("mixture", "guitar"):
                path = Path(row["paths"][name])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic audio identity")
        return rows

    def test_fixed_ids_paths_and_original_scores(self):
        report = self.build()
        self.assertEqual(report["split_counts"], {"train": 1, "validation": 1, "test": 1})
        rows = read_feature_manifest(self.manifest)
        self.assertEqual([row["track_id"] for row in rows], ["1", "4", "5"])
        self.assertEqual(rows[0]["annotations"], {"valence": .575, "arousal": .4})
        self.assertEqual(rows[0]["paths"]["mixture"], str(self.root / "audio/1.mp3"))
        self.assertEqual(rows[0]["paths"]["guitar"], str(self.root / "stems/1/guitar.mp3"))
        self.assertFalse(report["binary_labels_created"])
        with self.assertRaisesRegex(ValueError, "exists"):
            self.build()

    def test_metadata_dry_run_never_checks_audio_or_writes(self):
        report = self.build(dry_run=True)
        self.assertEqual(report["records"], 3)
        self.assertFalse(report["audio_files_checked"])
        self.assertFalse(self.manifest.exists())

    def test_duplicate_split_is_rejected(self):
        self.splits["validation"].write_text("1\n")
        with self.assertRaisesRegex(ValueError, "Repeated"):
            self.build(dry_run=True)

    def test_missing_annotation_is_not_silently_excluded(self):
        self.splits["test"].write_text("6\n")
        with self.assertRaisesRegex(ValueError, "lacks annotation"):
            self.build(dry_run=True)

    def test_unassigned_annotations_are_not_silently_excluded(self):
        self.annotations.write_text(self.annotations.read_text() + "6,0.1,0.2\n")
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            self.build(dry_run=True)

    def test_wrong_scale_and_unsafe_filename_are_rejected(self):
        self.annotations.write_text(self.annotations.read_text().replace("0.575", "5.6"))
        with self.assertRaisesRegex(ValueError, "0-1"):
            self.build(dry_run=True)
        self.annotations.write_text(self.annotations.read_text().replace("5.6", "0.575"))
        self.metadata.write_text(self.metadata.read_text().replace("1.mp3", "../1.mp3"))
        with self.assertRaisesRegex(ValueError, "Invalid PMEmo metadata"):
            self.build(dry_run=True)

    def test_manifest_rejects_dataset_confusion_and_traversal(self):
        self.build()
        row = read_feature_manifest(self.manifest)[0]
        for changes in ({"dataset": "mtg_jamendo"}, {"track_id": "track_0000001"},
                        {"relative_path": "../1.mp3"}, {"paths": {"mixture": "relative.mp3"}}):
            self.manifest.write_text(json.dumps({**row, **changes}) + "\n")
            with self.assertRaises(ValueError):
                read_feature_manifest(self.manifest)

    def test_presets_match_jamendo_scientific_settings(self):
        for method, role, jamendo in (("mert", "mert_segment_extraction", "mert"),
                                     ("chords_keys", "chord_key_extraction", "chords_keys")):
            pmemo, _ = load_configuration(default_config_path(f"experiments/pmemo/{method}.yaml"), role)
            reference, _ = load_configuration(default_config_path(f"experiments/mtg_jamendo/{jamendo}.yaml"), role)
            paths = {"manifest", "output_dir", "report"}
            self.assertEqual({k: v for k, v in pmemo.items() if k not in paths},
                             {k: v for k, v in reference.items() if k not in paths})

    def test_orchestration_matches_reference_except_dataset_provenance(self):
        for name, backend in (("extract_mert_segments", mert_segments), ("extract_chords_keys", chords_keys)):
            adapted = inspect.getsource(getattr(emotion_extraction, name))
            adapted = adapted.replace("mert_backend.", "").replace("harmony_backend.", "")
            adapted = adapted.replace("read_feature_manifest(", "read_segment_manifest(")
            adapted = adapted.replace('"dataset": "pmemo"', '"dataset": "mtg_jamendo"')
            adapted = adapted.replace("    settings.update(_adapter_identity())\n", "")
            adapted = adapted.replace('{**_settings(config), **_adapter_identity()}', '_settings(config)')
            self.assertEqual(ast.dump(ast.parse(adapted)), ast.dump(ast.parse(inspect.getsource(getattr(backend, name)))))

    def test_cli_defaults_and_overrides(self):
        parser = build_parser()
        for method in ("mert", "chords-keys"):
            args = parser.parse_args(["extract", "pmemo", method, "--source", "guitar", "--split", "test", "--dry-run"])
            self.assertEqual(args.source, ["guitar"])
            self.assertEqual(args.splits, ["test"])
            self.assertEqual(args.yaml_provenance["yaml_settings_before_cli_overrides"]["dataset"], "pmemo")

    def test_direct_entrypoint_help_from_outside_repository(self):
        directory = Path(__file__).resolve().parents[2] / "src/stemfusion/experiments/pmemo"
        for name in ("prepare_feature_manifest.py", "extract_mert.py", "extract_chords_keys.py"):
            result = subprocess.run([sys.executable, str(directory / name), "--help"], cwd=self.root,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_mert_real_writer_retains_tiny_tail_and_reuses(self):
        self.prepare_audio()
        model = self.root / "model"
        model.mkdir()
        for name in mert_segments.MODEL_FILES:
            (model / name).write_bytes(b"synthetic asset")
        config = mert_segments.MERTSegmentConfig(self.manifest, self.root / "mert", model_path=model,
                                                 device="cpu", splits=("train",))
        def extract(path):
            yield 0, 720000, np.ones((1, 12, 768), dtype=np.float32)
            yield 720000, 720001, np.ones((1, 12, 768), dtype=np.float32)
        factory = Mock(return_value=extract)
        report = emotion_extraction.extract_mert_segments(config, ("mixture", "guitar"), extractor_factory=factory)
        self.assertEqual((report["dataset"], report["extracted"], report["minimum_padded_segments"]), ("pmemo", 2, 2))
        marker = json.loads((config.output_dir / "mert_30s/1/segments.json").read_text())
        self.assertEqual(marker["segments"][-1]["minimum_padding_samples"], 399)
        factory.reset_mock()
        again = emotion_extraction.extract_mert_segments(config, ("mixture", "guitar"), extractor_factory=factory)
        self.assertEqual(again["reused"], 2)
        factory.assert_not_called()

    def test_extraction_dry_runs_do_not_load_models(self):
        self.build()
        mert = mert_segments.MERTSegmentConfig(self.manifest, self.root / "mert", dry_run=True)
        harmony = chords_keys.ChordKeyExtractionConfig(self.manifest, self.root / "harmony", self.root,
            self.root / "btc.yaml", self.root / "btc.pt", self.root / "report.json", dry_run=True)
        factory = Mock(side_effect=AssertionError("Model must not load"))
        self.assertEqual(emotion_extraction.extract_mert_segments(mert, ("mixture",), extractor_factory=factory)["planned"], 3)
        self.assertEqual(emotion_extraction.extract_chords_keys(harmony, runtime_factory=factory)["planned"], 18)
        self.assertFalse(harmony.report.exists())
        self.assertFalse(mert.output_dir.exists())

    def test_harmony_shared_writer_retries_only_failures_and_resumes(self):
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
                (output / "key.lab").write_text("C major\n")
                (output / "chords.mid").write_bytes(b"synthetic MIDI")
                return {"chord_intervals": 2, "midi_notes": 5, "key": "C major", "key_status": "estimated", "key_error": None}
        factory = Mock(return_value=Runtime())
        with patch.object(chords_keys, "_settings", return_value={"fixture": True}):
            report = emotion_extraction.extract_chords_keys(config, runtime_factory=factory)
            self.assertEqual((report["extracted"], report["repaired"], report["failed"], report["exit_code"]), (1, 1, 0, 0))
            before = (config.output_dir / "mixture/1/metadata.json").read_bytes()
            factory.reset_mock()
            again = emotion_extraction.extract_chords_keys(config, runtime_factory=factory)
            self.assertEqual(again["reused"], 2)
            factory.assert_not_called()
            self.assertEqual((config.output_dir / "mixture/1/metadata.json").read_bytes(), before)

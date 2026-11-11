from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from stemfusion.cli import build_parser
from stemfusion.configuration import ConfigurationError, default_config_path, load_configuration
from stemfusion.data.fma import (
    FMAManifestConfig,
    FMAManifestError,
    HTDEMUCS_SOURCES,
    build_fma_manifest,
    config_from_args,
    read_manifest,
)


class FMAManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.tracks_csv = self.root / "tracks.csv"
        self.mixture_root = self.root / "mixtures"
        self.source_root = self.root / "demucs"
        self.mixture_root.mkdir()
        self.source_root.mkdir()
        self.output_path = self.root / "fma_htdemucs.jsonl"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_metadata(self) -> None:
        with self.tracks_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["", "set", "set", "track"])
            writer.writerow(["", "split", "subset", "genre_top"])
            writer.writerow(["track_id", "", "", ""])
            writer.writerow([2, "training", "small", "Rock"])
            writer.writerow([3, "validation", "medium", "Classical"])
            writer.writerow([4, "test", "large", "Jazz"])

    def write_audio(self, track_id: str) -> None:
        folder = track_id[:3]
        mixture = self.mixture_root / folder / f"{track_id}.mp3"
        mixture.parent.mkdir(parents=True, exist_ok=True)
        mixture.write_bytes(b"mixture")
        for source in HTDEMUCS_SOURCES:
            stem = self.source_root / "htdemucs_6s" / track_id / f"{source}.mp3"
            stem.parent.mkdir(parents=True, exist_ok=True)
            stem.write_bytes(b"stem")

    def make_config(self, **overrides: object) -> FMAManifestConfig:
        values: dict[str, object] = {
            "tracks_csv": self.tracks_csv,
            "output_path": self.output_path,
            "mixture_root": self.mixture_root,
            "htdemucs_roots": (self.source_root,),
        }
        values.update(overrides)
        return FMAManifestConfig(**values)

    def test_builds_aligned_htdemucs_manifest(self) -> None:
        self.write_metadata()
        self.write_audio("000002")
        self.write_audio("000003")

        records, summary = build_fma_manifest(self.make_config())

        self.assertEqual([record.track_id for record in records], ["000002", "000003"])
        self.assertEqual([record.split for record in records], ["train", "validation"])
        self.assertEqual(summary.label_mapping, {"Classical": 0, "Rock": 1})
        self.assertEqual(summary.split_counts["train"], 1)
        self.assertEqual(summary.split_counts["validation"], 1)
        self.assertEqual(summary.split_counts["test"], 0)
        self.assertTrue(records[0].paths["drums"].endswith("htdemucs_6s/000002/drums.mp3"))
        self.assertEqual(read_manifest(self.output_path), records)

        metadata = json.loads(
            (self.root / "fma_htdemucs.metadata.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(metadata["sources"], sorted({"mixture", *HTDEMUCS_SOURCES}))

    def test_missing_audio_requires_explicit_aligned_drop(self) -> None:
        self.write_metadata()
        self.write_audio("000002")
        mixture = self.mixture_root / "000" / "000003.mp3"
        mixture.parent.mkdir(parents=True, exist_ok=True)
        mixture.write_bytes(b"mixture")

        with self.assertRaises(FMAManifestError):
            build_fma_manifest(self.make_config())

        records, summary = build_fma_manifest(
            self.make_config(allow_missing=True)
        )
        self.assertEqual([record.track_id for record in records], ["000002"])
        self.assertEqual(summary.dropped_tracks, 1)
        self.assertEqual(summary.missing_by_source, {source: 1 for source in HTDEMUCS_SOURCES})

    def test_supports_distributed_htdemucs_roots(self) -> None:
        self.write_metadata()
        first_root = self.root / "demucs_first"
        second_root = self.root / "demucs_second"
        first_root.mkdir()
        second_root.mkdir()

        for track_id, demucs_root in (
            ("000002", first_root),
            ("000003", second_root),
        ):
            folder = track_id[:3]
            mixture = self.mixture_root / folder / f"{track_id}.mp3"
            mixture.parent.mkdir(parents=True, exist_ok=True)
            mixture.write_bytes(b"mixture")
            for source in HTDEMUCS_SOURCES:
                stem = (
                    demucs_root
                    / "htdemucs_6s"
                    / track_id
                    / f"{source}.mp3"
                )
                stem.parent.mkdir(parents=True, exist_ok=True)
                stem.write_bytes(b"stem")

        partial = (
            first_root / "htdemucs_6s" / "000003" / "drums.mp3"
        )
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(b"incomplete duplicate")

        records, _ = build_fma_manifest(
            self.make_config(
                htdemucs_roots=(first_root, second_root),
            )
        )

        by_id = {record.track_id: record for record in records}
        self.assertEqual(
            set(by_id["000002"].paths),
            {"mixture", *HTDEMUCS_SOURCES},
        )
        self.assertTrue(
            by_id["000003"].paths["other"].startswith(
                str(second_root.resolve())
            )
        )
    def test_can_restrict_manifest_to_selected_track_ids(self) -> None:
        self.write_metadata()
        self.write_audio("000002")
        self.write_audio("000003")

        records, summary = build_fma_manifest(
            self.make_config(track_ids=("000003",))
        )

        self.assertEqual([record.track_id for record in records], ["000003"])
        self.assertEqual(records[0].label, 0)
        self.assertEqual(summary.metadata_tracks, 1)
        self.assertEqual(summary.written_tracks, 1)

    def manifest_yaml(self):
        return {"dataset": "fma_medium", "audio": {"mixture_extension": ".mp3"},
                "paths": {"tracks_csv": "tracks.csv", "mixture_root": "mixtures",
                          "htdemucs_root": ["demucs"], "output": "fma_htdemucs.jsonl"},
                "excluded_track_ids": ["000002"],
                "execution": {"allow_missing": False, "overwrite": False, "dry_run": False}}

    def test_explicit_exclusions_preserve_label_indices_and_record_drops(self):
        self.write_metadata()
        self.write_audio("000003")
        records, summary = build_fma_manifest(self.make_config(excluded_track_ids=("000002",)))
        self.assertEqual([record.track_id for record in records], ["000003"])
        self.assertEqual(records[0].label, 0)
        self.assertEqual(summary.label_mapping, {"Classical": 0, "Rock": 1})
        self.assertEqual((summary.metadata_tracks, summary.written_tracks, summary.dropped_tracks), (2, 1, 1))
        self.assertEqual(summary.excluded_track_ids, ["000002"])
        self.assertEqual(summary.missing_by_source, {})
        saved = json.loads(self.output_path.with_suffix(".report.json").read_text())
        self.assertEqual(saved["excluded_track_ids"], ["000002"])

    def test_unknown_duplicate_or_selected_exclusions_are_rejected(self):
        self.write_metadata()
        for values in ({"excluded_track_ids": ("999999",)},
                       {"excluded_track_ids": ("000002", "000002")},
                       {"excluded_track_ids": ("000002",), "track_ids": ("000002",)}):
            with self.subTest(values=values), self.assertRaises(FMAManifestError):
                build_fma_manifest(self.make_config(**values))

    def test_yaml_resolves_relative_paths_and_cli_replaces_roots_and_exclusions(self):
        path = self.root / "manifest.yaml"
        path.write_text(yaml.safe_dump(self.manifest_yaml()))
        parser = build_parser()
        args = parser.parse_args(["data", "fma-manifest", "--config", str(path)])
        config = config_from_args(args)
        self.assertEqual(config.tracks_csv, self.tracks_csv)
        self.assertEqual(config.mixture_root, self.mixture_root)
        self.assertEqual(config.htdemucs_roots, (self.source_root,))
        self.assertEqual(config.output_path, self.output_path)
        self.assertEqual(config.excluded_track_ids, ("000002",))
        alternative = self.root / "other_demucs"
        override = parser.parse_args(["data", "fma-manifest", "--config", str(path),
            "--tracks-csv", str(self.root / "alternate.csv"), "--htdemucs-root", str(alternative),
            "--exclude-track-id", "3", "--allow-missing", "--dry-run"])
        config = config_from_args(override)
        self.assertEqual(config.tracks_csv, self.root / "alternate.csv")
        self.assertEqual(config.htdemucs_roots, (alternative,))
        self.assertEqual(config.excluded_track_ids, ("000003",))
        self.assertTrue(config.allow_missing)
        self.assertTrue(config.dry_run)
        self.assertIn("tracks_csv", override.yaml_provenance["cli_overrides"])
        self.assertEqual(config_from_args(parser.parse_args(["data", "fma-manifest", "--config", str(path)])).excluded_track_ids, ("000002",))

    def test_yaml_rejects_invalid_exclusions_unknown_keys_and_wrong_dataset(self):
        path = self.root / "manifest.yaml"
        for key, value in (("excluded_track_ids", [2]), ("excluded_track_ids", ["000002", "000002"]),
                           ("dataset", "gtzan"), ("unexpected", True)):
            values = self.manifest_yaml()
            values[key] = value
            path.write_text(yaml.safe_dump(values))
            with self.subTest(key=key, value=value), self.assertRaises(ConfigurationError):
                load_configuration(path, "fma_manifest")
        values = self.manifest_yaml()
        values["execution"]["allow_missing"] = "false"
        path.write_text(yaml.safe_dump(values))
        with self.assertRaises(ConfigurationError):
            load_configuration(path, "fma_manifest")

    def test_direct_script_loads_yaml_without_pythonpath_and_preserves_existing_manifest(self):
        self.write_metadata()
        self.write_audio("000003")
        path = self.root / "manifest.yaml"
        path.write_text(yaml.safe_dump(self.manifest_yaml()))
        script = Path(__file__).resolve().parents[2] / "src/stemfusion/experiments/fma_medium/prepare_manifest.py"
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        command = [sys.executable, str(script), "--config", str(path)]
        generated = subprocess.run(command, cwd=self.root, env=env, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(generated.stdout)["written_tracks"], 1)
        before = self.output_path.read_bytes()
        dry = subprocess.run([*command, "--dry-run"], cwd=self.root, env=env, capture_output=True, text=True, check=True)
        self.assertTrue(json.loads(dry.stdout)["dry_run"])
        refused = subprocess.run(command, cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(refused.returncode, 2)
        self.assertIn("already exists", refused.stdout)
        self.assertEqual(self.output_path.read_bytes(), before)

    def test_default_direct_and_unified_parsers_share_the_same_preset(self):
        args = build_parser().parse_args(["data", "fma-manifest"])
        config = config_from_args(args)
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(config.tracks_csv, root / "data/manifests/fma_medium/metadata/tracks.csv")
        self.assertEqual(config.output_path, root / "data/manifests/fma_medium/fma_medium_htdemucs.jsonl")
        self.assertEqual(config.excluded_track_ids, ())
        self.assertTrue(config.allow_missing)
        self.assertFalse(config.overwrite)
        self.assertEqual(Path(args.yaml_provenance["config_files"][0]["path"]), default_config_path("experiments/fma_medium/manifest.yaml"))

    def test_default_preset_filters_by_file_availability_and_supports_strict_override(self):
        self.write_metadata()
        self.write_audio("000002")
        arguments = ["data", "fma-manifest", "--tracks-csv", str(self.tracks_csv),
                     "--mixture-root", str(self.mixture_root), "--htdemucs-root", str(self.source_root),
                     "--output", str(self.output_path), "--dry-run"]
        parser = build_parser()
        config = config_from_args(parser.parse_args(arguments))
        records, summary = build_fma_manifest(config)
        self.assertEqual([record.track_id for record in records], ["000002"])
        self.assertEqual(summary.excluded_track_ids, [])
        self.assertEqual((summary.metadata_tracks, summary.written_tracks, summary.dropped_tracks), (2, 1, 1))
        self.assertEqual(summary.missing_by_source, {source: 1 for source in {"mixture", *HTDEMUCS_SOURCES}})
        self.assertFalse(self.output_path.exists())
        strict = config_from_args(parser.parse_args([*arguments, "--no-allow-missing"]))
        self.assertFalse(strict.allow_missing)
        with self.assertRaisesRegex(FMAManifestError, "Configured audio is incomplete"):
            build_fma_manifest(strict)

if __name__ == "__main__":
    unittest.main()

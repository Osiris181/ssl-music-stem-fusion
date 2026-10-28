from __future__ import annotations

import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path

from stemfusion.training.genre import (
    GenreTrainingConfig,
    GenreTrainingError,
    SourceFeatureDirectory,
    canonicalize_sources,
    normalize_feature_cache,
    plan_genre_training,
    resolve_feature_cache_path,
    train_genre_classifier,
)


TRAINING_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "torch", "torchmetrics")
)


class GenreTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.manifest = self.root / "manifest.jsonl"
        self.features = self.root / "features"
        self.output = self.root / "output"
        self.features.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_manifest(self) -> None:
        records = []
        for split_index, split in enumerate(("train", "validation", "test")):
            for label, genre in ((0, "Classical"), (1, "Rock")):
                track_id = f"{split_index * 2 + label + 1:06d}"
                records.append(
                    {
                        "track_id": track_id,
                        "split": split,
                        "genre": genre,
                        "label": label,
                        "paths": {"guitar": f"/audio/{track_id}.wav"},
                    }
                )
        self.manifest.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def write_caches(self, sources: tuple[str, ...], valid_pickle: bool) -> None:
        split_ids = {
            "train": ("000001", "000002"),
            "validation": ("000003", "000004"),
            "test": ("000005", "000006"),
        }
        for source_index, source in enumerate(sources):
            for split, track_ids in split_ids.items():
                path = self.features / f"fma_musicfm_{source}_{split}.pkl"
                if valid_pickle:
                    cache = {
                        track_id: [
                            float(source_index + 1),
                            float(int(track_id) % 2),
                            0.5,
                        ]
                        for track_id in track_ids
                    }
                    with path.open("wb") as handle:
                        pickle.dump(cache, handle)
                else:
                    path.write_bytes(b"cache placeholder")

    def make_config(self, **overrides: object) -> GenreTrainingConfig:
        values: dict[str, object] = {
            "manifest_path": self.manifest,
            "feature_directory": self.features,
            "output_directory": self.output,
            "sources": ("guitar",),
            "expected_feature_dimension": 3,
            "hidden_dimensions": (4, 2),
            "dropout": 0.0,
            "batch_size": 2,
            "epochs": 1,
            "device": "cpu",
        }
        values.update(overrides)
        return GenreTrainingConfig(**values)

    def test_canonicalizes_fusion_order_and_legacy_names(self) -> None:
        sources = canonicalize_sources(
            ("vocal", "drum", "mix", "electric_guitar")
        )

        self.assertEqual(
            sources,
            ("mixture", "guitar", "drums", "vocals"),
        )

    def test_normalizes_clean_and_legacy_cache_keys(self) -> None:
        cache = normalize_feature_cache(
            {
                "000002": [1.0],
                "fma/000/000003_electric_guitar.wav": [2.0],
            }
        )

        self.assertEqual(set(cache), {"000002", "000003"})
        with self.assertRaises(GenreTrainingError):
            normalize_feature_cache(
                [("000002", [1.0]), ("000002_drum.wav", [2.0])]
            )

    def test_dry_run_resolves_canonical_caches_without_ml_packages(self) -> None:
        self.write_manifest()
        self.write_caches(("guitar", "drums"), valid_pickle=False)

        plan = plan_genre_training(
            self.make_config(sources=("drums", "guitar"))
        )

        self.assertEqual(plan["sources"], ["guitar", "drums"])
        self.assertEqual(plan["input_dimension"], 6)
        self.assertEqual(plan["number_of_classes"], 2)
        self.assertEqual(plan["manifest_split_counts"]["test"], 2)

    def test_explicit_source_directory_enables_legacy_cache_names(self) -> None:
        self.write_manifest()
        legacy = self.root / "guitar_classifier"
        legacy.mkdir()
        for split in ("train", "validation", "test"):
            (legacy / f"fma_musicfm_{split}.pkl").write_bytes(b"legacy")
        config = self.make_config(
            source_feature_directories=(
                SourceFeatureDirectory("guitar", legacy),
            )
        )

        path = resolve_feature_cache_path(config, "guitar", "train")

        self.assertEqual(path, legacy / "fma_musicfm_train.pkl")

    @unittest.skipUnless(
        TRAINING_DEPENDENCIES_AVAILABLE,
        "Optional training dependencies are not installed.",
    )
    def test_tiny_mid_fusion_training_writes_reproducibility_artifacts(self) -> None:
        self.write_manifest()
        self.write_caches(("guitar", "drums"), valid_pickle=True)

        result = train_genre_classifier(
            self.make_config(sources=("drums", "guitar"))
        )

        self.assertEqual(result["sources"], ["guitar", "drums"])
        self.assertEqual(result["input_dimension"], 6)
        self.assertEqual(result["split_examples"]["test"], 2)
        for filename in (
            "checkpoint.pt",
            "data_manifest_checksum.txt",
            "environment.json",
            "metrics.json",
            "predictions.npz",
            "resolved_config.json",
            "training_history.json",
        ):
            self.assertTrue((self.output / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()

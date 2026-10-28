from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from stemfusion.models.late_fusion import (
    RMSWeightedLateFusionMLP,
    UnweightedLateFusionMLP,
)
from stemfusion.training._htdemucs_late_fusion import (
    LateFusionTrainingConfig,
    plan_late_fusion_training,
    train_late_fusion_classifier,
)


class HTDemucsLateFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.manifest_path = self.root / "manifest.jsonl"
        self.feature_directory = self.root / "musicfm"
        self.energy_directory = self.root / "rms"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_inputs(self) -> None:
        records = []
        track_number = 1
        for split in ("train", "validation", "test"):
            for label, genre in enumerate(("Blues", "Rock")):
                track_id = f"{track_number:06d}"
                track_number += 1
                records.append(
                    {
                        "track_id": track_id,
                        "split": split,
                        "genre": genre,
                        "label": label,
                        "paths": {
                            "mixture": f"/audio/{track_id}.mp3",
                            "drums": f"/audio/{track_id}/drums.mp3",
                        },
                    }
                )
        self.manifest_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        for split in ("train", "validation", "test"):
            split_records = [record for record in records if record["split"] == split]
            for source, offset in (("mixture", 0.0), ("drums", 1.0)):
                features = {
                    record["track_id"]: np.full(
                        4,
                        record["label"] + offset,
                        dtype=np.float32,
                    )
                    for record in split_records
                }
                self.write_pickle(
                    self.feature_directory / f"fma_musicfm_{source}_{split}.pkl",
                    features,
                )
                energies = {
                    record["track_id"]: np.asarray(
                        [0.5 + record["label"] + offset],
                        dtype=np.float32,
                    )
                    for record in split_records
                }
                self.write_pickle(
                    self.energy_directory / f"fma_rms_{source}_{split}.pkl",
                    energies,
                )

    @staticmethod
    def write_pickle(path: Path, data: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(data, handle)

    def make_config(self, *, weighted: bool, dry_run: bool = False) -> LateFusionTrainingConfig:
        return LateFusionTrainingConfig(
            manifest_path=self.manifest_path,
            feature_directory=self.feature_directory,
            output_directory=self.root / ("weighted" if weighted else "unweighted"),
            sources=("mixture", "drums"),
            weighted=weighted,
            energy_feature_directory=self.energy_directory if weighted else None,
            expected_feature_dimension=4,
            hidden_dimensions=(8, 4),
            batch_size=2,
            epochs=1,
            device="cpu",
            dry_run=dry_run,
        )

    def test_unweighted_model_sums_independent_logits(self) -> None:
        model = UnweightedLateFusionMLP(
            number_of_sources=2,
            feature_dimension=4,
            number_of_classes=2,
            hidden_dimensions=(4, 2),
            dropout=0.0,
        )
        features = torch.ones((3, 2, 4))
        expected = sum(
            classifier(features[:, index, :])
            for index, classifier in enumerate(model.stem_classifiers)
        )

        self.assertTrue(torch.allclose(model(features), expected))

    def test_weighted_model_applies_one_rms_gate_per_source(self) -> None:
        model = RMSWeightedLateFusionMLP(
            number_of_sources=2,
            feature_dimension=4,
            number_of_classes=2,
            hidden_dimensions=(4, 2),
            dropout=0.0,
            energy_hidden_dimension=2,
        )
        for gate in model.energy_gates:
            for parameter in gate.parameters():
                parameter.data.zero_()
        features = torch.ones((3, 2, 4))
        energies = torch.ones((3, 2, 1))
        expected = 0.5 * sum(
            classifier(features[:, index, :])
            for index, classifier in enumerate(model.stem_classifiers)
        )

        self.assertTrue(torch.allclose(model(features, energies), expected))

    def test_dry_run_resolves_separate_musicfm_and_rms_caches(self) -> None:
        self.write_inputs()

        result = plan_late_fusion_training(
            self.make_config(weighted=True, dry_run=True)
        )

        self.assertTrue(result["dry_run"])
        self.assertEqual(result["fusion"], "rms_weighted_late")
        self.assertTrue(
            result["energy_caches"]["mixture"]["train"].endswith(
                "fma_rms_mixture_train.pkl"
            )
        )

    def test_dry_run_resolves_legacy_appended_energy_caches(self) -> None:
        self.write_inputs()
        for split in ("train", "validation", "test"):
            for source in ("mix", "drums"):
                self.write_pickle(
                    self.energy_directory
                    / f"fma_musicfm_{source}_{split}_with_energy.pkl",
                    {"000001": np.zeros(1025, dtype=np.float32)},
                )

        result = plan_late_fusion_training(
            replace(
                self.make_config(weighted=True, dry_run=True),
                legacy_appended_energy=True,
            )
        )

        self.assertTrue(
            result["energy_caches"]["mixture"]["train"].endswith(
                "fma_musicfm_mix_train_with_energy.pkl"
            )
        )

    def test_unweighted_and_weighted_training_write_artifacts(self) -> None:
        self.write_inputs()

        unweighted = train_late_fusion_classifier(self.make_config(weighted=False))
        weighted = train_late_fusion_classifier(self.make_config(weighted=True))

        self.assertEqual(unweighted["fusion"], "unweighted_late")
        self.assertEqual(weighted["fusion"], "rms_weighted_late")
        for name in ("checkpoint.pt", "metrics.json", "predictions.npz"):
            self.assertTrue((self.root / "unweighted" / name).is_file())
            self.assertTrue((self.root / "weighted" / name).is_file())


if __name__ == "__main__":
    unittest.main()

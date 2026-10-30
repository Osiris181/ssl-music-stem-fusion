from __future__ import annotations

import contextlib
import io
import json
import os
import pickle
import statistics
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from stemfusion.cli import build_parser
from stemfusion.configuration import default_config_path, load_configuration
from stemfusion.data.features import FeatureDataError
from stemfusion.data.fma_features import SOURCES, load_fma_features
from stemfusion.experiments.fma_medium.training import FMAExperimentConfig, config_from_args, run_experiment, save_stem_results
from stemfusion.training.late_fusion import assemble_ensemble, branch_seed
from stemfusion.training.metrics import GenreMetrics
from stemfusion.training.mid_fusion import match_capacity, model_parameter_count, train_mid_condition
from stemfusion.training.reporting import METRICS, summarize_results
from stemfusion.training.trainer import TrainingSettings

SEEDS = (32, 64, 128, 256, 512)


class TrainingProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.old_threads)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.jsonl"
        self.feature_dir = self.root / "features"
        self.output = self.root / "outputs"
        self.feature_dir.mkdir()
        records = []
        for split_index, split in enumerate(("train", "validation", "test")):
            for label, genre in enumerate(("Blues", "Rock")):
                records.append({"track_id": f"{split_index * 2 + label + 1:06d}", "split": split,
                                "genre": genre, "label": label, "paths": {"mixture": "/unused.mp3"}})
        self.manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
        for index, source in enumerate(SOURCES):
            for split in ("train", "validation", "test"):
                cache = {record["track_id"]: np.asarray([record["label"], index / 6, 0.1, -0.1], dtype=np.float32)
                         for record in records if record["split"] == split}
                self.write_cache(source, split, cache)
        self.data = load_fma_features(self.manifest, self.feature_dir, 4, 2, require_reports=False)
        self.settings = TrainingSettings(hidden_dimensions=(8, 4), batch_size=2, epochs=1, dropout=0.2, device="cpu")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_cache(self, source, split, values) -> None:
        with (self.feature_dir / f"fma_musicfm_{source}_{split}.pkl").open("wb") as handle:
            pickle.dump(values, handle)

    def config(self, family="mid_fusion", **changes) -> FMAExperimentConfig:
        _, provenance = load_configuration(default_config_path(f"experiments/fma_medium/{family}.yaml"), family)
        config = FMAExperimentConfig(self.manifest, self.feature_dir, self.output, family,
                                     provenance["experiment_metadata"]["runs"], SEEDS,
                                     settings=self.settings, dimension=4, number_of_classes=2)
        return replace(config, **changes)

    def train(self, directory, **changes):
        arguments = dict(data=self.data, sources=("mixture",), settings=self.settings, seed=32,
                         output_directory=directory, task=GenreMetrics(2), provenance={})
        arguments.update(changes)
        return train_mid_condition(**arguments)

    def test_capacity_widths_and_actual_counts(self) -> None:
        for count, width, parameters in ((4, 3071, 4728336), (5, 3753, 5776912), (6, 4435, 6825488)):
            result = match_capacity(1024, 16, count, (1024, 512))
            self.assertEqual(result["hidden_dimensions"], [width, 512])
            self.assertEqual(result["fusion_parameters"], parameters)
            difference = abs(result["parameter_difference"])
            for neighbor in (width - 1, width + 1):
                self.assertLessEqual(difference, abs(model_parameter_count(1024, 16, (neighbor, 512)) - parameters))

    def test_zero_vectors_retained_for_every_condition(self) -> None:
        self.write_cache("guitar", "train", {"000001": np.zeros(4, dtype=np.float32), "000002": np.ones(4, dtype=np.float32)})
        data = load_fma_features(self.manifest, self.feature_dir, 4, 2, require_reports=False)
        self.assertEqual(data.splits["train"].track_ids, ("000001", "000002"))
        self.assertEqual(data.audit["caches"]["guitar/train"]["zero_vectors"], 1)

    def write_reports(self, dimension=4, counts=2) -> None:
        for source in SOURCES:
            report = {"source": source, "layer_index": 7, "expected_dimension": dimension,
                      "sample_rate": 24000, "temporal_pooling": "mean", "max_seconds": None,
                      "failed": 0, "exit_code": 0, "dry_run": False,
                      "checkpoint_path": "/model/pretrained_msd.pt", "stats_path": "/model/msd_stats.json",
                      "split_summaries": [{"split": split, "selected": counts, "failed": 0} for split in ("train", "validation", "test")]}
            (self.feature_dir / f"extraction_report_musicfm_{source}.json").write_text(json.dumps(report))

    def test_extraction_reports_reject_wrong_layer_checkpoint_and_counts(self) -> None:
        self.write_reports()
        data = load_fma_features(self.manifest, self.feature_dir, 4, 2)
        self.assertEqual(len(data.audit["extraction_reports"]), 6)
        path = self.feature_dir / "extraction_report_musicfm_piano.json"
        original = json.loads(path.read_text())
        for update in ({"layer_index": 12}, {"checkpoint_path": "/model/pretrained_fma.pt"}, {"failed": 1},
                       {"split_summaries": [{"split": "train", "selected": 1, "failed": 0}]}):
            path.write_text(json.dumps({**original, **update}))
            with self.assertRaises(FeatureDataError):
                load_fma_features(self.manifest, self.feature_dir, 4, 2)

    def test_missing_extra_wrong_shape_dtype_and_nonfinite_features_rejected(self) -> None:
        original = {"000001": np.ones(4, dtype=np.float32), "000002": np.ones(4, dtype=np.float32)}
        invalid = [{"000001": original["000001"]}, {**original, "000099": original["000001"]},
                   {**original, "000001": np.ones((1, 4), dtype=np.float32)},
                   {**original, "000001": np.ones(4, dtype=np.float64)},
                   {**original, "000001": np.full(4, np.nan, dtype=np.float32)}]
        for cache in invalid:
            with self.subTest(cache=cache):
                self.write_cache("vocals", "test", cache)
                with self.assertRaises(FeatureDataError):
                    load_fma_features(self.manifest, self.feature_dir, 4, 2, require_reports=False)

    def test_duplicate_ids_and_class_coverage_rejected(self) -> None:
        text = self.manifest.read_text()
        self.manifest.write_text(text + text.splitlines()[0] + "\n")
        with self.assertRaises(ValueError):
            load_fma_features(self.manifest, self.feature_dir, 4, 2, require_reports=False)
        self.manifest.write_text("\n".join(line for line in text.splitlines() if '000006' not in line) + "\n")
        with self.assertRaises(FeatureDataError):
            load_fma_features(self.manifest, self.feature_dir, 4, 2, require_reports=False)

    def test_full_matrix_and_seed_reproducibility(self) -> None:
        mid = run_experiment(self.config(), data=self.data)
        late = run_experiment(self.config("late_fusion"), data=self.data)
        self.assertEqual(mid["seeds"], list(SEEDS))
        self.assertEqual(len(mid["conditions"]), 7)
        self.assertEqual(len(late["conditions"]), 6)
        self.assertEqual(len(mid["paired_differences"]), 3)
        self.assertEqual(len(late["paired_differences"]), 3)
        self.assertEqual(len(late["standalone_stems"]), 5)
        standalone_path = self.output / "summary_stem_only.json"
        standalone = json.loads(standalone_path.read_text())
        self.assertEqual(standalone["analysis_role"], "supporting")
        self.assertFalse(standalone["new_training_performed"])
        self.assertFalse(standalone["new_inference_performed"])
        self.assertFalse(standalone["checkpoint_reselection_performed"])
        self.assertEqual(standalone["seeds"], list(SEEDS))
        self.assertEqual(set(standalone["conditions"]), {f"{source}_only" for source in SOURCES[1:]})
        saved_accuracies = []
        for seed in SEEDS:
            branch = self.output / "branches" / f"seed_{seed}" / "guitar" / "metrics.json"
            record = json.loads(branch.read_text())
            saved_accuracies.append(record["test"]["accuracy"])
            self.assertEqual(standalone["runs"]["guitar_only"][str(seed)]["branch_seed"], record["seed"])
        self.assertEqual(standalone["conditions"]["guitar_only"]["accuracy"]["mean"], statistics.mean(saved_accuracies))
        self.assertEqual(standalone["conditions"]["guitar_only"]["accuracy"]["sample_sd"], statistics.stdev(saved_accuracies))
        text = standalone_path.with_suffix(".txt").read_text()
        self.assertIn("Guitar only", text)
        self.assertIn("Mixture only", text)
        self.assertIn("supporting analysis", text)
        self.assertNotIn("Study context:", text)
        for family in ("mid_fusion", "late_fusion"):
            report = self.output / f"summary_{family}.txt"
            text = report.read_text()
            self.assertIn("Mean +/- sample SD", text)
            self.assertIn("Paired differences", text)
            self.assertIn("M = original mixture", text)
            self.assertNotIn("Study context:", text)
        for seed in SEEDS:
            branches = list((self.output / "branches" / f"seed_{seed}").glob("*/checkpoint.pt"))
            self.assertEqual(len(branches), 11)
        self.assertEqual(len(list(self.output.rglob("checkpoint.pt"))), 85)
        example = mid["runs"]["matched_mixture_4"][32]
        self.assertEqual(example["model"]["input_dimension"], 4)
        self.assertEqual(example["model"]["sources"], ["mixture"])
        self.assertEqual(example["trainable_parameters"], example["capacity_matching"]["matched_parameters"])
        ensemble = late["runs"]["mixture_guitar_drums_piano"][32]
        self.assertEqual(len({member["seed"] for member in ensemble["members"]}), 4)
        self.assertEqual(ensemble["aggregation"], "sum_logits_then_softmax")
        with np.load(Path(ensemble["output_directory"]) / "predictions.npz") as saved:
            expected = sum(np.load(Path(member["checkpoint_path"]).parent / "predictions.npz")["logits"] for member in ensemble["members"])
            np.testing.assert_array_equal(saved["logits"], expected)
            np.testing.assert_allclose(saved["probabilities"].sum(axis=1), 1)
        pair = mid["paired_differences"]["mixture_guitar_drums_piano"]
        values = [mid["runs"]["mixture_guitar_drums_piano"][seed]["test"]["accuracy"] - mid["runs"]["matched_mixture_4"][seed]["test"]["accuracy"] for seed in SEEDS]
        self.assertEqual(pair["metrics"]["accuracy"]["mean"], statistics.mean(values))
        self.assertEqual(pair["metrics"]["accuracy"]["sample_sd"], statistics.stdev(values))
        checkpoint_paths = list(self.output.rglob("checkpoint.pt"))
        before = {path: path.read_bytes() for path in checkpoint_paths}
        with patch("torch.optim.Adam", side_effect=AssertionError("Completed fits must be reused without training.")):
            repeated_mid = run_experiment(self.config(), data=self.data)
            repeated_late = run_experiment(self.config("late_fusion"), data=self.data)
        self.assertEqual(repeated_mid["conditions"], mid["conditions"])
        self.assertEqual(repeated_late["conditions"], late["conditions"])
        self.assertEqual(before, {path: path.read_bytes() for path in checkpoint_paths})

    def test_repeated_seed_matches_checkpoint_and_predictions(self) -> None:
        first = self.train(self.root / "first")
        self.train(self.root / "second")
        with np.load(self.root / "first/predictions.npz") as a, np.load(self.root / "second/predictions.npz") as b:
            np.testing.assert_array_equal(a["logits"], b["logits"])
        first_weights = torch.load(self.root / "first/checkpoint.pt", weights_only=False)["model_state_dict"]
        second_weights = torch.load(self.root / "second/checkpoint.pt", weights_only=False)["model_state_dict"]
        self.assertTrue(all(torch.equal(value, second_weights[name]) for name, value in first_weights.items()))
        reused = self.train(self.root / "first", resume=True)
        self.assertEqual(first, reused)
        with self.assertRaises(ValueError):
            self.train(self.root / "first", seed=64, resume=True)
        (self.root / "first/predictions.npz").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.train(self.root / "first", resume=True)

    def test_automatic_reuse_rejects_incompatible_corrupt_and_incomplete_runs(self) -> None:
        config = self.config(selected_run="mixture_only", seeds=(32,))
        run_experiment(config, data=self.data)
        directory = self.output / "branches/seed_32/mixture_1"
        checkpoint = (directory / "checkpoint.pt").read_bytes()
        with self.assertRaisesRegex(ValueError, "incompatible settings"):
            run_experiment(replace(config, settings=replace(self.settings, batch_size=4)), data=self.data)
        predictions = directory / "predictions.npz"
        predictions.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            run_experiment(config, data=self.data)
        (directory / "run_manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            run_experiment(config, data=self.data)
        self.assertEqual((directory / "checkpoint.pt").read_bytes(), checkpoint)
        # Explicit overwrite overrides automatic reuse and repairs the incomplete fit.
        repaired = run_experiment(replace(config, overwrite=True), data=self.data)
        self.assertEqual(repaired["number_of_runs"], 1)
        self.assertTrue((directory / "run_manifest.json").is_file())

    def test_independent_mixture_branches_have_different_weights(self) -> None:
        self.train(self.root / "one", seed=branch_seed(32, 0))
        self.train(self.root / "two", seed=branch_seed(32, 1))
        first = torch.load(self.root / "one/checkpoint.pt", weights_only=False)["model_state_dict"]
        second = torch.load(self.root / "two/checkpoint.pt", weights_only=False)["model_state_dict"]
        self.assertTrue(any(not torch.equal(value, second[name]) for name, value in first.items()))

    def test_ensemble_rejects_prediction_alignment_mismatch(self) -> None:
        first = self.train(self.root / "one", seed=32)
        second = self.train(self.root / "two", seed=64)
        path = self.root / "two/validation_predictions.npz"
        with np.load(path) as saved:
            arrays = {name: saved[name].copy() for name in saved.files}
        arrays["track_ids"] = arrays["track_ids"][::-1]
        np.savez(path, **arrays)
        with self.assertRaisesRegex(ValueError, "identical track IDs"):
            assemble_ensemble(branches=[first, second], output_directory=self.output, task=GenreMetrics(2), outer_seed=32, condition="test")

    def test_validation_only_selects_checkpoint_and_test_is_evaluated_last(self) -> None:
        task = GenreMetrics(2)
        calls = []
        original = task.evaluate

        def evaluate(logits, labels, *, full=True):
            calls.append((labels is self.data.splits["test"].labels, full))
            return original(logits, labels, full=full)

        with patch.object(task, "evaluate", side_effect=evaluate), patch.object(task, "selection_score", return_value=1.0):
            result = self.train(self.output, task=task, settings=replace(self.settings, epochs=2))
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(calls, [(False, False), (False, False), (False, True), (True, True)])

    def test_summary_rejects_incomplete_seeds_and_mismatched_data(self) -> None:
        metric = {"test": {name: 0.5 for name in METRICS}, "data_fingerprint": "a"}
        with self.assertRaises(ValueError):
            summarize_results({"a": {32: metric}}, (32, 64), {})
        with self.assertRaises(ValueError):
            summarize_results({"a": {32: metric}, "b": {32: {**metric, "data_fingerprint": "b"}}}, (32,), {"a": "b"})

    def test_stem_reporting_never_trains_when_branches_are_missing_or_incompatible(self) -> None:
        config = self.config("late_fusion", seeds=(32,))
        run_experiment(config, data=self.data)
        with patch("torch.optim.Adam", side_effect=AssertionError("Reporting must not train.")), patch(
            "stemfusion.experiments.fma_medium.training.load_fma_features", side_effect=AssertionError("Reporting must not load features.")
        ):
            summary = save_stem_results(config)
            self.assertEqual(len(summary["conditions"]), 5)
            with self.assertRaisesRegex(ValueError, "Completed branch required"):
                save_stem_results(replace(config, seeds=SEEDS))
        self.assertFalse((self.output / "summary_stem_only.json").exists())
        directory = self.output / "branches/seed_32/guitar"
        self.train(directory, sources=("guitar",), seed=branch_seed(32, 1),
                   settings=replace(self.settings, dropout=0.0), overwrite=True)
        with self.assertRaisesRegex(ValueError, "share data and training/evaluation settings"):
            save_stem_results(config)

    def test_dry_run_and_validate_only_do_not_train_or_write_outputs(self) -> None:
        with patch("stemfusion.training.trainer.fit_classifier", side_effect=AssertionError("training forbidden")):
            plan = run_experiment(self.config(dry_run=True))
            audit = run_experiment(self.config(validate_only=True), data=self.data)
        self.assertEqual(len(plan["conditions"]), 7)
        self.assertTrue(audit["validation_only"])
        self.assertFalse(self.output.exists())

    def test_branch_seeds_are_stable_distinct_and_pairable(self) -> None:
        for seed in SEEDS:
            derived = [branch_seed(seed, index) for index in range(6)]
            self.assertEqual(derived[0], seed)
            self.assertEqual(len(set(derived)), 6)
            self.assertEqual(derived, [branch_seed(seed, index) for index in range(6)])

    def test_cli_defaults_use_one_shared_profile_and_seed_subsets(self) -> None:
        parser = build_parser()
        mid = config_from_args(parser.parse_args(["train", "fma-medium", "mid-fusion"]))
        late = config_from_args(parser.parse_args(["train", "fma-medium", "late-fusion"]))
        self.assertEqual(mid.settings, late.settings)
        self.assertTrue(mid.resume)
        self.assertTrue(late.resume)
        overwritten = config_from_args(parser.parse_args(["train", "fma-medium", "late-fusion", "--overwrite"]))
        self.assertTrue(overwritten.overwrite)
        self.assertFalse(overwritten.resume)
        self.assertEqual(mid.seeds, SEEDS)
        self.assertEqual(mid.output_directory, late.output_directory)
        self.assertEqual(mid.manifest_path, late.manifest_path)
        self.assertEqual(mid.summary_directory, mid.output_directory.parent)
        redirected = config_from_args(parser.parse_args(["train", "fma-medium", "mid-fusion", "--output-dir", str(self.output)]))
        self.assertEqual(redirected.summary_directory, self.output)
        separate = config_from_args(parser.parse_args(["train", "fma-medium", "mid-fusion", "--output-dir", str(self.output), "--summary-dir", str(self.root / "reports")]))
        self.assertEqual(separate.summary_directory, self.root / "reports")
        subset = config_from_args(parser.parse_args(["train", "fma-medium", "mid-fusion", "--seed", "32", "--run", "matched_mixture_4"]))
        self.assertEqual(subset.seeds, (32,))
        self.assertEqual(subset.selected_run, "matched_mixture_4")

    def test_direct_scripts_work_outside_checkout(self) -> None:
        project = Path(__file__).resolve().parents[2]
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        for script in ("train_mid.py", "train_late.py"):
            result = subprocess.run([sys.executable, str(project / "src/stemfusion/experiments/fma_medium" / script), "--help"],
                                    cwd=self.root, env=environment, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--resume", result.stdout)

    def test_direct_cli_trains_one_real_multiclass_synthetic_condition(self) -> None:
        records = []
        for split_index, split in enumerate(("train", "validation", "test")):
            cache = {}
            for label in range(16):
                track_id = f"{split_index * 16 + label + 1:06d}"
                records.append({"track_id": track_id, "split": split, "genre": f"class_{label}",
                                "label": label, "paths": {"mixture": "/unused.mp3"}})
                cache[track_id] = np.full(4, label / 16, dtype=np.float32)
            for source in SOURCES:
                self.write_cache(source, split, cache)
        self.manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
        self.write_reports(counts=16)
        project = Path(__file__).resolve().parents[2]
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        script = project / "src/stemfusion/experiments/fma_medium/train_mid.py"
        result = subprocess.run([sys.executable, str(script), "--manifest", str(self.manifest),
            "--feature-dir", str(self.feature_dir), "--output-dir", str(self.output), "--run", "mixture_only",
            "--seed", "32", "--epochs", "1", "--hidden-dimensions", "8", "4", "--device", "cpu",
            "--expected-feature-dimension", "4"], cwd=self.root, env=environment, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Mid fusion test results", result.stdout)
        self.assertIn("Mixture only", result.stdout)
        self.assertIn("Single-seed", result.stdout)
        self.assertIn("Detailed results:", result.stdout)
        self.assertTrue((self.output / "branches/seed_32/mixture_1/checkpoint.pt").is_file())
        self.assertTrue((self.output / "summary_mid_fusion_mixture_only_seeds_32.json").is_file())
        text_report = self.output / "summary_mid_fusion_mixture_only_seeds_32.txt"
        readable = text_report.read_text()
        self.assertIn("Single-seed", readable)
        self.assertIn("Mixture only", readable)
        self.assertIn("Detailed JSON: summary_mid_fusion_mixture_only_seeds_32.json", readable)
        self.assertIn(f"Readable report: {text_report}", result.stdout)
        late_script = project / "src/stemfusion/experiments/fma_medium/train_late.py"
        common = ["--manifest", str(self.manifest), "--feature-dir", str(self.feature_dir),
                  "--output-dir", str(self.output), "--seed", "32", "--epochs", "1",
                  "--hidden-dimensions", "8", "4", "--device", "cpu", "--expected-feature-dimension", "4"]
        late = subprocess.run([sys.executable, str(late_script), *common], cwd=self.root,
                              env=environment, text=True, capture_output=True, timeout=30)
        self.assertEqual(late.returncode, 0, late.stderr)
        self.assertIn("Supporting analysis: individual stem classifiers", late.stdout)
        checkpoints = {path: path.read_bytes() for path in self.output.rglob("checkpoint.pt")}
        report_only = subprocess.run([sys.executable, str(late_script), *common, "--report-stems-only"],
                                     cwd=self.root, env=environment, text=True, capture_output=True, timeout=30)
        self.assertEqual(report_only.returncode, 0, report_only.stderr)
        self.assertIn("Stem only test results", report_only.stdout)
        self.assertIn("Guitar only", report_only.stdout)
        self.assertTrue((self.output / "summary_stem_only_seeds_32.json").is_file())
        self.assertEqual(checkpoints, {path: path.read_bytes() for path in self.output.rglob("checkpoint.pt")})


if __name__ == "__main__":
    unittest.main()

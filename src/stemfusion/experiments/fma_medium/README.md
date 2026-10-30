# FMA-Medium genre classification

This dataset uses frozen MusicFM-MSD layer-7 embeddings of the mixture and
five HT-Demucs stems to predict 16 genre classes. The existing splits and
class mapping are preserved; `other` is excluded.

| File | Purpose |
| --- | --- |
| `prepare_manifest.py` | Runnable manifest builder from official metadata and aligned audio paths. |
| `extraction.py` | Runnable MusicFM extraction interface. |
| `train_mid.py` | Runnable mid-fusion matrix, including capacity-matched mixture controls. |
| `train_late.py` | Runnable independently trained late-fusion matrix, including matched mixture ensembles. |
| `training.py` | FMA-specific parser, dataset handoff, condition validation, seed scheduling, branch reuse, and summary assembly. |
| `stem_results.py` | Verify completed source branches and aggregate their standalone metrics without training, inference, or checkpoint reselection. |

Shared training functions are in `training/`, model architectures in
`models/`, and strict feature loading in `data/fma_features.py`.
Each public file supports direct execution without PYTHONPATH.

## Configuration

- MusicFM settings and checkout location: `configs/features/musicfm_msd_layer7.yaml`.
- Extraction manifest/output/source list: `configs/experiments/fma_medium/extraction.yaml`.
- Shared classifier settings and manifest/feature/output paths: `configs/training/fma_medium_genre.yaml`.
- Conditions: `configs/experiments/fma_medium/mid_fusion.yaml` and `late_fusion.yaml`.

CLI paths override YAML paths. Relative YAML paths resolve against the YAML
file; relative CLI paths resolve against the current working directory.

## Manifest preparation

From this directory:

```bash
python prepare_manifest.py \
  --tracks-csv /path/to/fma_metadata/tracks.csv \
  --mixture-root /path/to/fma_medium \
  --htdemucs-root /path/to/htdemucs_outputs
```

The HT-Demucs output root must contain `htdemucs_6s/`. Repeat
`--htdemucs-root` for tracks distributed across disks. The script calls the
shared builder in `data/fma.py` and defaults to
`data/manifests/fma_medium/fma_medium_htdemucs.jsonl` in this checkout.
`--output` overrides that path. Input CLI paths are relative to the working
directory; the default output is independent of it. Use `--allow-missing` only
to explicitly exclude incomplete tracks, `--dry-run` to inspect without writing,
and `--overwrite` to replace an existing manifest. Existing manifests can be
reused directly for extraction and training.

## Training commands

From this directory, inspect the plans and validate feature contents:

```bash
python train_mid.py --dry-run
python train_late.py --dry-run
python train_mid.py --validate-only
```

To run the complete matrices:

```bash
python train_mid.py
python train_late.py
```

Full late-fusion runs also generate standalone Guitar/Drums/Piano/Bass/Vocals
summaries from the selected branch results. To generate those summaries from
an already completed experiment without loading audio or feature caches:

```bash
python train_late.py --report-stems-only
```

This reads existing branch artifacts only and checks their completion hashes,
seeds, training settings, labels, and validation/test prediction alignment.
Missing or incompatible branches cause an error; reporting never starts
training to fill them in. Use `--seed 32` for a single-seed report.

These commands train over seeds `32, 64, 128, 256, 512`. Without `--run`,
each runs its whole condition matrix. The default common output root is
`outputs/fma_medium/phase2/`. The late command can reuse the standard
mixture model trained by the mid command when their effective settings match.

To run one condition and one seed:

```bash
python train_mid.py --run matched_mixture_4 --seed 32
```

The same workflows are available from the project root:

```bash
PYTHONPATH=src python -m stemfusion train fma-medium mid-fusion
PYTHONPATH=src python -m stemfusion train fma-medium late-fusion
```

## Experiment matrix

Mid fusion has seven conditions: the standard mixture MLP; three wider
mixture-only controls; and `M+G+D+P`, `M+G+D+P+B`, `M+G+D+P+B+V`.
The measured matched first-hidden widths are 3071, 3753, 4435 for
1024-dimensional features and the standard 1024/512 architecture.

Late fusion has six conditions: three 4/5/6-member independent mixture-only
ensembles, and the same three source combinations. The ordinary mixture
reference is reported alongside them. Every branch is trained and selected
independently; selected logits are summed equally, then softmax is applied.

Both workflows require exactly aligned, finite float32 source caches. Valid
zero vectors are retained. Training uses Adam 1e-4, batch size 128, 20 epochs,
ReLU and dropout 0.2. Selection maximizes validation micro accuracy + macro
one-vs-rest ROC-AUC + macro average precision, on the 0–1 scale. Ties retain
the earliest checkpoint. Test metrics never choose checkpoints.

## Artifacts and reuse

Each classifier saves its checkpoint, effective config, environment, epoch
history, validation/test predictions, metrics, and completion checksums.
Ensembles save selected branch references and aggregated predictions.
Data audits record cache/manifest hashes, class mapping, shapes, and zero counts.
Summary JSON files contain per-seed results, means, sample SD (ddof=1), and
paired fusion-minus-matched-control differences. A single-seed subset has
`sample_sd: null`, not a five-seed claim.

When the command finishes, it also prints a concise test-results table,
paired differences for included controls, and the saved summary path.
The readable table is also saved to a `.txt` file beside the detailed JSON,
including settings, source abbreviations, and metric explanations. Git allows
`outputs/**/summary_*.txt` and `outputs/**/summary_*.json` to be committed for reporting; other generated
artifacts remain ignored.
Complete reports are `outputs/fma_medium/summary_mid_fusion.txt`/`.json`
and `summary_late_fusion.txt`/`.json`. The shared YAML's `paths.summary_dir`
sets that location, separately from the artifact root. Condition/seed subsets
keep descriptive filename suffixes. An explicit `--output-dir` redirects
summaries too unless `--summary-dir` is supplied.

Supporting standalone results are saved as `summary_stem_only.txt`/`.json`
in the same directory. They include all five source classifiers, the shared
mixture-only reference, and the actual outer/branch seeds. The late-fusion
summary also includes a standalone-stem table when the all-five-stems
condition is run. Standalone scores describe each source's predictive
effectiveness; the matched fusion comparisons evaluate incremental value.

Completed, compatible runs are verified and reused automatically. `--resume`
remains accepted but is not required. Use `--overwrite` explicitly to retrain
and replace existing outputs. Interrupted individual
fits restart from scratch with a new output location or `--overwrite`;
optimizer-state continuation is not implemented.

Only synthetic training checks have been run during implementation. Historical
FMA test scores influenced the selected nested stem sets; these reruns are
not a newly independent confirmatory holdout.

See [the full training guide](../../../../docs/fma_htdemucs_training.md) and
[the design](../../../../context/PHASE2_EXPERIMENTAL_DESIGN.md).

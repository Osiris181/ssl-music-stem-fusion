# FMA-Medium genre classification

This dataset uses frozen MusicFM-MSD layer-7 embeddings of the mixture and
five HT-Demucs stems to predict 16 genre classes. The existing splits and
class mapping are preserved; `other` is excluded.

| File | Purpose |
| --- | --- |
| `prepare_manifest.py` | Runnable manifest builder from official metadata and aligned audio paths. |
| `separation.py` | Runnable HT-Demucs separation using the shared backend and dataset paths. |
| `extraction.py` | Runnable MusicFM extraction interface. |
| `extraction_cnn.py` | Phase 3 frozen MSD_musicnn extraction with the same backend and window policy as GTZAN and ISMIR04. |
| `train_mid_cnn.py` | Train the same seven CNN mid-fusion conditions using the shared genre trainer and five seeds. |
| `train_late_cnn.py` | Train independent CNN late-fusion branches, reuse completed CNN-M, and report matched mixture ensembles plus standalone stems. |
| `train_mid.py` | Runnable mid-fusion matrix, including repeated-mixture input controls. |
| `train_late.py` | Runnable independently trained late-fusion matrix, including matched mixture ensembles. |
| `stem_results.py` | Compatibility interface to shared saved-branch analysis in `experiments/_stem_results.py`. |

Shared training functions are in `training/`, model architectures in
`models/`, and strict feature loading in `data/fma_features.py`.
Both training launchers call `experiments/_genre.py` directly, just like
GTZAN and ISMIR04; no dataset-level training wrapper is needed.
Each public file supports direct execution without PYTHONPATH.

## Configuration

- Manifest CSV, audio roots, and output: `configs/experiments/fma_medium/manifest.yaml`.
- MusicFM settings and checkout location: `configs/features/musicfm_msd_layer7.yaml`.
- HT-Demucs paths: `configs/experiments/fma_medium/separation.yaml`; inference settings inherit `configs/separators/htdemucs_6s.yaml`.
- Extraction manifest/output/source list: `configs/experiments/fma_medium/extraction.yaml`.
- Common genre hyperparameters inherited by FMA-Medium, GTZAN, and ISMIR04: `configs/training/genre.yaml`.
- Dataset paths, class count, fusion/model selection, and conditions: `configs/experiments/fma_medium/mid_fusion.yaml` and `late_fusion.yaml`; both inherit the shared genre profile directly.

CLI paths override YAML paths. Relative YAML paths resolve against the YAML
file; relative CLI paths resolve against the current working directory.

## Source separation

From this directory, inspect or run separation with:

```bash
python separation.py --dry-run
python separation.py
```

The script loads `configs/experiments/fma_medium/separation.yaml` and calls
the shared HT-Demucs implementation. Existing complete six-stem folders are
reused automatically. `--input-root`, `--output-root`, and repeated
`--existing-output-root` arguments select different storage locations.
Directory discovery reads filenames under the input roots; use these
commands only when separation is needed.

## Manifest preparation

From this directory:

```bash
python prepare_manifest.py --dry-run
python prepare_manifest.py
```

Both this script and `stemfusion data fma-manifest` automatically load
`configs/experiments/fma_medium/manifest.yaml`. The default CSV is
`data/manifests/fma_medium/metadata/tracks.csv`; place your official copy there
or change `paths.tracks_csv`. The copied `genres.csv` is reference information
and is not required by the builder. Edit the audio roots for your storage.

The HT-Demucs output root must contain `htdemucs_6s/`. Multiple roots can be
listed in YAML or supplied with repeated `--htdemucs-root` flags. Explicit
CLI lists replace YAML lists. For example:

```bash
python prepare_manifest.py \
  --tracks-csv /path/to/fma_metadata/tracks.csv \
  --mixture-root /path/to/fma_medium \
  --htdemucs-root /path/to/htdemucs_outputs \
  --output /path/to/fma_medium_manifest.jsonl
```

The script calls the shared metadata/path alignment logic in `data/fma.py`
and preserves the official split assignments and full 16-class label mapping.
The preset builds a common track set from the configured audio files.
Use `--no-allow-missing` to require complete audio availability.

The default output remains
`data/manifests/fma_medium/fma_medium_htdemucs.jsonl`. YAML paths resolve
relative to the declaring file; CLI paths resolve relative to the working
directory. Dry runs check audio-file availability without writing a manifest;
they can take time on HDD storage. Existing manifests are protected unless
`--overwrite` is explicit. Reuse the existing manifest for completed extraction
and training; this configuration addition does not require rebuilding it.

## Phase 3 CNN extraction

`extraction_cnn.py` uses the shared `experiments/_cnn_extraction.py` orchestration
and `features/musicnn.py` encoder. It loads
`configs/experiments/fma_medium/extraction_cnn.yaml` and reuses the existing
`data/manifests/fma_medium/fma_medium_htdemucs.jsonl`, preserving six-digit IDs,
the 16-class mapping, original partitions, and mixture/stem audio paths.
No new manifest or separation pass is needed for the existing setup.

Activate the [isolated musicnn environment](../../../../../environments/musicnn/README.md),
then run from this experiment directory:

```bash
python extraction_cnn.py --dry-run
python extraction_cnn.py
```

The script configures legacy Keras and local CUDA paths automatically. It uses
the official frozen standard MSD_musicnn checkpoint, mono 16 kHz audio and
official log-mel preprocessing. Consecutive 187-frame windows have no overlap;
the last partial window is retained and right-padded with silence. Average all
200-dimensional window embeddings into one float32 vector per track/source.
This is the same explicit tail adaptation used in the other CNN datasets.

The shared [audio-decoding guard](../../../../../docs/cnn_audio_decoding.md)
retries invalid decodes with FFmpeg and revalidates references for incomplete
tracks. Wrong-length cached views are repaired with backups; healthy caches
remain unchanged. True source corruption/mismatches still fail explicitly.

The sources are mixture, guitar, drums, piano, bass and vocals. Resampled stems
are aligned in memory to the mixture length within a 0.25-second tolerance;
larger differences fail extraction. The full recording is retained, valid
silence is retained, and no additional normalization or harmonic features are
added. CNN inference uses batches of 16 windows and a YAML GPU memory cap of
4096 MB, overridable with `--gpu-memory-mb N`. Full-track waveform and
spectrogram preprocessing uses CPU memory.

Caches save to `data/features/fma_medium/msd_musicnn/{split}/{source}/{track_id}.npz`.
Each file contains `embedding` (float32, shape 200) and JSON `metadata`, including
the integer genre label, source/audio identity, checkpoint and implementation
hashes, sample/frame counts, and window/padding policy. `numpy.load` can read
these files with `allow_pickle=False`.

The tqdm bar counts selected source views. Final readable counts point to
`outputs/fma_medium/cnn/extraction_report.json`. Reruns reuse matching caches;
changed settings or input metadata cause a compatibility error and require
explicit `--overwrite` to replace caches. Failed views are reported with a
nonzero exit status; they do not redefine the intended cohort.

For a bounded check with separate test output:

```bash
python extraction_cnn.py --limit-per-split 1 \
  --output-dir /tmp/fma_medium_cnn_check/features \
  --report /tmp/fma_medium_cnn_check/report.json --fail-fast
```

CNN caches are separate from the MusicFM caches consumed by `train_mid.py` and
`train_late.py`. The MSD pretraining overlap audit remains unresolved.

The FMA CNN training loader automatically retains only tracks with all six
source embeddings. A track missing any source is skipped from every CNN
condition, including mixture-only and repeated-mixture controls. No CLI policy
option or track-ID list is needed. The data audit records skipped tracks,
missing sources, and actual split counts; the original manifest and caches are
not modified. Existing embeddings still require valid labels, provenance,
finite vectors and aligned source windows. Unknown cache IDs, empty retained
splits and loss of a genre class remain errors. A dry-run plan reports manifest
counts; validation and training report the retained feature counts.

### CNN mid-fusion training

After extraction is complete, activate the main PyTorch training environment
and run from this FMA-Medium directory:

```bash
python train_mid_cnn.py --validate-only
python train_mid_cnn.py --dry-run
python train_mid_cnn.py
```

The entry point calls `_cnn_genre.py` and the same `_genre.py` scheduler, MLPs,
trainer, evaluation and reporting used by GTZAN and ISMIR04. Configuration is
`configs/experiments/fma_medium/mid_fusion_cnn.yaml`, which inherits the existing
FMA-Medium `mid_fusion.yaml` and shared `configs/training/genre.yaml` settings.
The FMA adapter preserves six-digit IDs and the existing 16-class genre mapping.
It uses the same retained complete-view tracks for every condition, finite float32 vectors,
matching extraction provenance and source-aligned windows. Silent vectors remain
valid, and no normalization is added. Training reads cached features only.

The seven conditions are mixture only; repeated mixture x4/x5/x6; and M+G+D+P,
then +B, then +V. Their input widths are 200/800/1000/1200, followed by hidden
layers 1024/512 and 16 output logits. Corresponding repeated-mixture and fusion
models have identical nominal parameter counts. ReLU, dropout 0.2, unweighted
cross-entropy, Adam 1e-4, batch 128 and 20 epochs match the original genre
experiments. There is no early stopping. The best checkpoint maximizes the
validation sum of accuracy, macro ROC-AUC and macro AP, retaining the earliest
epoch on ties. Test evaluation follows model selection.

The five seeds remain 32, 64, 128, 256 and 512: 35 classifier fits in the full
matrix. Results print after completion and save as
`outputs/fma_medium/cnn/summary_mid_fusion.txt` and `.json`, including means,
sample SDs and paired fusion-minus-control differences. Checkpoints and per-run
metrics save under `outputs/fma_medium/cnn/runs/`. Reruns reuse verified completed
fits; `--overwrite` explicitly retrains them. `--run NAME` and repeated `--seed N`
support subsets with distinct summary filenames. An explicit `--output-dir`
redirects summaries too unless `--summary-dir` is supplied. No CNN late-fusion
or standalone stem-only fits are included in this matrix.

### CNN late-fusion training

Use the main PyTorch training environment and run from this directory:

```bash
python train_late_cnn.py --dry-run
python train_late_cnn.py --validate-only
python train_late_cnn.py
```

`train_late_cnn.py` calls `_cnn_genre.py`, then the same `_genre.py` scheduler
and `training/late_fusion.py` branch fitter/ensemble evaluator as MusicFM.
`configs/experiments/fma_medium/late_fusion_cnn.yaml` inherits the existing
MusicFM `late_fusion.yaml` and shared `configs/training/genre.yaml` recipe.
Each branch is **200 -> 1024 -> 512 -> 16**, with biased linear layers,
ReLU, dropout 0.2, unweighted cross-entropy, Adam 1e-4, batch 128, and 20
epochs. Checkpoint selection and metrics are the same as CNN mid fusion.

For each outer seed (32, 64, 128, 256, 512), M1 is the completed CNN mixture-only
checkpoint under `outputs/fma_medium/cnn/runs/branches/seed_N/mixture_1/`.
The command checks and reads M1 without retraining or modifying it. Train
M2–M6 independently on the same mixture embedding, and G/D/P/B/V on their
own embeddings. The existing position-paired SHA256 seeds match MusicFM late
fusion. This gives **50 new branch fits** over the five outer seeds.

Reuse those branches to report M1+G+D+P, then +B, then +V against independent
mixture ensembles of four, five, and six members. Both sum raw logits equally,
then softmax/argmax. Full runs also report Guitar/Drums/Piano/Bass/Vocals alone
from the selected stem checkpoints, without extra fitting. The intentional FMA
complete-view track skip is identical to CNN mid fusion.

New checkpoints and per-ensemble predictions save beneath
`outputs/fma_medium/cnn/late_fusion/runs/`. The terminal and
`outputs/fma_medium/cnn/summary_late_fusion.{txt,json}` include the mixture
reference, six ensembles, five standalone stems, and paired differences with
sample SD. A companion `summary_stem_only.{txt,json}` is saved in the same
directory. Subsets use the same seed/run filename suffixes as other genre runs.

`--validate-only` checks the retained CNN cache and all requested M1 references
without fitting or writing outputs. `--run NAME` and repeated `--seed N`
select subsets. `--reference-output-dir` selects another completed CNN mid-run
root; `--output-dir` selects the new late-run root and also redirects summaries
unless `--summary-dir` is supplied. Reference and late-run roots must be
separate, non-nested directories. Reruns reuse verified complete late fits;
`--overwrite` replaces only late outputs, never the CNN-M reference. A missing,
changed, or incompatible reference stops the run rather than creating a new M1.

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
python train_mid.py --run repeated_mixture_4 --seed 32
```

The same workflows are available from the project root:

```bash
PYTHONPATH=src python -m stemfusion train fma-medium mid-fusion
PYTHONPATH=src python -m stemfusion train fma-medium late-fusion
```

## Experiment matrix

Mid fusion has seven conditions: the standard mixture MLP;
`M+M+M+M`, `M+M+M+M+M`, `M+M+M+M+M+M`; and
`M+G+D+P`, `M+G+D+P+B`, `M+G+D+P+B+V`. All retain hidden widths
1024/512. Repeated-mixture controls exactly match their corresponding fusion
input dimensions and parameter counts, without adding new input information.
They are named `repeated_mixture_4`, `repeated_mixture_5`, and `repeated_mixture_6`.
The superseded widened-control results and previous mid summaries are archived
under `outputs/fma_medium/archive/widened_mixture_20261103/`. Unchanged fusion
and branch artifacts remain in their existing run directories for verified reuse.

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

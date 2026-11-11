# Jamendo single-dataset Music2Emo reproduction

This is a reproduction **attempt** of the original-mixture, single-dataset
MERT + chord/key baseline, before adding stem comparisons.
The published single-dataset reference is macro PR-AUC **0.1521** and macro
ROC-AUC **0.7806**. The paper's 0.1543 / 0.7810 result instead uses all four
datasets with multitask learning and distillation; it is not this experiment's
target. Existing author checkpoints are not loaded for training.

Sources are the supplied `Music2Emotion/Music2Emo.pdf` (implementation details,
Table II / III) and the author release
[`c5f7984f78d9928b841f25bd05a47377c12c58ca`](https://github.com/AMAAI-Lab/Music2Emotion/tree/c5f7984f78d9928b841f25bd05a47377c12c58ca).
The adapted classifier retains the upstream MIT notice in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Protocol

| Setting | Value |
| --- | --- |
| Dataset / inputs | Provided Jamendo split-0, 56 mood/theme tags; original mixture only |
| Audio features | Frozen cached MERT-v1-95M, saved-array indices `[5,6]` concatenated into 1,536 values |
| Training augmentation | Conditional-uniform random start, then random end; average that consecutive segment range |
| Validation / test pooling | Mean of every segment; no stochastic augmentation |
| Harmonic scope | Whole-track normalized chords and key from the same original mixture |
| Chord input | First 100 root/quality tokens, zero-padded; root and quality vocabularies each use 14 embeddings of width 4 |
| Chord encoder | Sinusoidal positions, zero CLS, 2 Transformer layers, 8 heads, feedforward width 64, dropout 0.1; no padding mask, matching release |
| Key input | Scalar mode, major=0/minor=1; unavailable `None` maps to 0 as in the released loader |
| Classifier | Concatenate 1,536 audio + 8 learned chord + 1 key values; `1545 -> 512 -> 256 -> 56`, ReLU hidden activations |
| Loss | Author frequency-weighted BCEWithLogitsLoss; `pos_weight=1/p`, `weight=2/(1+pos_weight)` |
| Optimizer | Adam, learning rate `1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay 0 |
| Batch / budget | Batch size 8, maximum **200 epochs**, no scheduler |
| Early stopping | Released configuration: validation weighted BCE, mode `min`, patience 10 validation checks, `min_delta=0.0001`; validate every epoch |
| Checkpoint selection | Minimum validation weighted BCE, earliest epoch on exact ties |
| Checkpoint retention | Five lowest validation-loss checkpoints plus the last; `best.pt` aliases the selected state |
| Seeds | **32, 64, 128, 256, 512**, identical splits and hyperparameters |
| Test reporting | Sigmoid outputs; macro AUROC and macro average precision (called PR-AUC in the release) |

The paper specifies 200 epochs; the released YAML says up to 500 with early
stopping. This experiment retains the chosen **paper epoch cap** of 200 but
uses the released early-stopping configuration. This is an explicit combination,
not an exact copy of the release's 500-epoch limit. The paper does not specify
checkpoint selection or early stopping. The released single-task entry also
comments out construction of its early-stopping callback while still referencing
it, so the implemented rule follows its declared configuration rather than
claiming that this broken entry verifies the authors' actual stopping behavior.
Test predictions are
computed only after training and reloading the validation-selected state.
There is no test-driven checkpoint, seed, threshold, or hyperparameter selection.

Early stopping and checkpoint selection have separate improvement rules:
the patience counter resets only when loss decreases by **more than 0.0001**
relative to its last significant-improvement score. Checkpoint ranking still
accepts **any strictly lower loss**, without a minimum delta. Ten consecutive
non-improving validation checks stop training; this cannot happen before epoch
11 with the default settings. The final test uses the lowest-loss checkpoint,
not necessarily the last significant-improvement epoch or stopping epoch.

Missing key estimates remain `None` in saved extraction files. Their scalar
mode fallback is a **training-loader convention from the author**, not a new
estimated key. Unknown chord qualities raise an error rather than silently
dropping intervals or tracks. Full chord IDs are unused by the released
classifier; its active inputs are root/quality tokens and the scalar mode.

## Configuration and required data

The direct command loads
`configs/experiments/mtg_jamendo/reproduction.yaml`. Paths resolve
relative to that file; CLI values override supported YAML defaults. Fixed
architecture/protocol fields are strictly checked, not silently ignored.

The dataset preset now inherits `configs/training/mood_theme_tagging.yaml`,
which inherits `configs/training/music2emo.yaml`. This layout-only refactor
preserves the original resolved settings and artifact locations. Historical
saved reports may name the former `mert_chord_key/reproduction.yaml`; those
provenance records are not rewritten. The chosen budget remains 200 epochs
and checkpoint selection remains minimum validation weighted BCE, not maximum
validation PR-AUC.

Required local inputs:

- The shared Jamendo JSONL manifest and original 183-entry `tag_list.npy`.
- Cached mixture MERT directories under `<mert_dir>/mert_30s/<bucket>/<id>/`.
- Harmony directories under `<harmony_dir>/mixture/<bucket>/<id>/`.
- `chord_root.json`, `chord_attr.json`, and `probas_train.npy` from the pinned
  author's `dataset/jamendo/meta/` directory. The configured local copies live
  under `data/manifests/mtg_jamendo/music2emotion/` and remain ignored data.
  Other users must copy these three files or point `paths.author_metadata` to
  their author checkout's metadata directory. No separate local-path YAML is
  required. The provided prevalences must match the training labels/tag order.

Preparation verifies completion checksums, shapes, IDs, sources, splits,
finite values, label coverage, and class frequencies. It never opens original
or separated audio. Arrays are loaded once into RAM and reused across seeds;
selected audio features take approximately 0.8 GiB before Python/worker overhead.
No MERT or BTC model weights are loaded during downstream training.

From the repository root, install the downstream dependencies into the chosen
environment if needed:

```bash
uv pip install -e '.[music2emo]'
```

From `src/stemfusion/experiments/mtg_jamendo/`:

```bash
python train_reproduction.py --dry-run
python train_reproduction.py --validate-only
python train_reproduction.py
```

`--dry-run` prints the plan without loading feature files or instantiating a
model. `--validate-only` reads all cached inputs without fitting classifiers
or writing experiment outputs. The last command runs all five seeds in order
on one configured device. The equivalent installed command is
`stemfusion train mtg-jamendo author-reproduction`.

Repeat `--seed` to select a subset; it replaces the YAML list. For example,
`--seed 32` trains/reuses that seed only. A partial seed selection writes a
distinct summary filename, so it cannot replace the five-seed summary.
`--device cpu` and `--num-workers 0` are available. CLI budget overrides are
recorded; changing `--epochs` away from 200 labels the summary as a non-paper
budget. `--epochs 500` explicitly selects the released maximum instead of the
paper cap. Early stopping remains active. Patience/minimum delta are defined
in YAML; `--early-stopping-patience` and `--early-stopping-min-delta` are optional
overrides recorded in the run signature. Do not mix different stopping settings
within a directly comparable experiment matrix.

## Checkpoints, resume, and results

Per-seed output:

```text
outputs/mtg_jamendo/mert_chord_key/author_reproduction_early_stopping/seed_<seed>/
├── checkpoints/epoch_NNNN.pt     # Up to five validation-loss-ranked states
├── last.pt                      # Model, optimizer, epoch, best state, history, patience, ranking
├── best.pt                      # Selected model, published after training completes
├── history.json                 # Per-epoch training and validation loss
├── resolved_config.json
├── predictions_validation.npz   # Track IDs, targets, logits, probabilities
├── predictions_test.npz
├── result.json                  # Selected epoch, per-tag/macro metrics, parameters, provenance
└── run_manifest.json            # Completed-run signature and artifact checksums
```

Resume is enabled by default. A verified completed seed is reused without
refitting. An interrupted run resumes at the next epoch from `last.pt`, using
epoch-local deterministic seeds; an interrupted partial epoch is repeated.
The significant-improvement score and patience counter are restored, not reset.
If a run stopped before its final reports were written, resume finalizes that
checkpoint without training another epoch. Retained ranked files are verified
before continuing. Superseded ranked checkpoints are pruned as training proceeds.
Unknown, incompatible, or corrupted outputs fail rather than being silently
accepted. `--overwrite` explicitly restarts instead of reusing checkpoints;
do not add it for ordinary continuation. Use one writer per output directory.
The earlier fixed-budget partial run remains untouched in `author_reproduction/`;
the new default directory prevents accidental cross-protocol reuse.

After all requested seeds finish, the command prints results and writes:

```text
outputs/mtg_jamendo/summary_author_reproduction.json
outputs/mtg_jamendo/summary_author_reproduction.txt
```

The JSON retains per-seed details; the TXT includes completed/selected epochs,
stopping reasons, test scores, **mean +/- sample SD**, and the published reference.
Overridden stopping settings are labeled explicitly. Both summaries match
the repository's Git-visible `summary_*` rules. Predictions enable later
confidence-interval analyses without retraining. SD measures training-run
variability, not statistical significance or track-level sampling uncertainty.

## Explicit reproducibility differences and limits

This is not a claim of exact paper-result equivalence:

- Five fixed seeds, strict deterministic algorithms, epoch-local sampling
  seeds, and math attention are our reproducibility additions. Worker processes
  restart each epoch rather than using the released persistent workers, making
  interrupted-run resume deterministic. Sampling distributions are unchanged.
- Segment files are ordered **numerically** to follow the paper's consecutive
  temporal windows. The released loader uses lexicographic filename sorting,
  which differs for tracks with at least 11 segments.
- This runner is **single-device with an effective batch size of 8**. The paper
  reports four V100 GPUs; the released DDP code uses a per-device batch of 8.
  We do not silently claim identical global-batch or hardware behavior.
- The installed Torch/package versions are recorded. The released training
  entry sets matmul precision to `medium`; this runner records and retains that
  setting. Matching seed values alone cannot guarantee equality across different
  hardware, numerical kernels, software versions, or preprocessing histories.
- The feature extraction's documented minimum padding adaptations remain:
  MERT tails below 400 samples and the repaired one-sample BTC CQT tails.
  The default released MIDI policy, including its single-interval edge case,
  has not been silently changed.

Only the downstream classifier is trained; frozen audio embeddings remain
unchanged. This baseline is isolated from the completed genre experiments.
Before adding stems, choose one consistent training/selection protocol for all
compared baseline, stem-fusion, and matched-control conditions. Do not compare
200-epoch-cap validation-loss-selected baselines against 500-epoch PR-AUC-selected
stem models as though their protocols matched.

# FMA-Medium genre classification

This is the only active dataset experiment. It uses the original mixture and
HT-Demucs stems with frozen MusicFM representations for 16-class genre
classification. Only downstream classifiers are trained.

## Files and responsibilities

| File | Purpose |
| --- | --- |
| `extraction.py` | Owns the FMA extraction parser, default task YAML, and config construction; calls shared MusicFM extraction. |
| `train_mid.py` | Dataset entry point for a single source or concatenated source features; delegates to the shared genre trainer. |
| `train_late.py` | Dataset entry point for equal-sum source logits; delegates to the retained shared late-fusion trainer. |

Shared engines live in [training](../../training/README.md), not in copied
dataset training loops. Configuration matrices are in
[configs/experiments/fma_medium](../../../../configs/experiments/fma_medium/README.md).

The public tasks are grouped in this directory:

```text
fma_medium/
├── extraction.py
├── train_mid.py
├── train_late.py
└── README.md
```

## Inputs and commands

Extraction consumes an aligned audio manifest and the local MusicFM checkout
with MSD weights/statistics. Training consumes the manifest and source/split
MusicFM caches. Cache names are `fma_musicfm_<source>_<split>.pkl`;
splits are `train`, `validation`,
and `test`. The default feature width is 1024. Explicit legacy cache-directory
adapters remain available; no existing features or splits were changed.

From the repository root, inspect options with:

```bash
python src/stemfusion/experiments/fma_medium/extraction.py --help
PYTHONPATH=src python -m stemfusion extract fma-medium --help
PYTHONPATH=src python -m stemfusion train fma-medium mid-fusion --help
PYTHONPATH=src python -m stemfusion train fma-medium late-fusion --help
```

`extraction.py` supports direct Python-file execution without `PYTHONPATH`;
it uses the same parser and YAML settings as the package command.

Each command loads its standard YAML preset automatically. Use `--config`
to select an experiment file and `--run` to select one named training
condition; explicit CLI settings override YAML. See
[configuration usage](../../../../docs/configuration.md) for full examples.

For extraction, the default task YAML is
`configs/experiments/fma_medium/extraction.yaml`. FMA-only overrides belong
there; its shared feature profile provides the base MusicFM settings. Numerical
parser defaults are not duplicated in Python.

One extraction command processes the YAML `sources` list automatically, with
one encoder reused across sources. Use repeated `--source` options to override
that list. Individual caches remain source/split-specific; a batch report
summarizes completion/failures. No shell loop is required.

For a mixture-only reference, select only `--source mixture` in the mid-fusion
command. To plan a mixture-plus-stem run, use:

```bash
PYTHONPATH=src python -m stemfusion train fma-medium mid-fusion \
  --manifest data/manifests/fma_htdemucs.jsonl \
  --feature-dir data/features/fma_medium/musicfm_msd_layer7 \
  --source mixture --source guitar --source drums --source piano \
  --output-dir outputs/fma_medium/mid_fusion/mixture_guitar_drums_piano/seed42 \
  --dry-run
```

Use `late-fusion` in place of `mid-fusion` and a separate output directory
for the other retained runner. Dry runs validate the manifest/cache paths
without loading feature contents or starting training. Use a new MSD/layer-7
output directory rather than the copied legacy caches. The extraction workflow
is in [the extraction guide](../../../../docs/fma_medium_extraction.md).

## Retained behavior versus revised design

This organization step preserves the existing MLP architectures, Adam
learning rate `1e-4`, batch size 128, 20-epoch budget, dropout 0.2,
hidden widths 1024/512, and seed-42 defaults. Comparable settings are shared;
the final five-seed list is not yet fixed.

The current late-fusion engine optimizes one model containing separate source
MLPs through a summed-logit loss, and selects a checkpoint for that whole
model. It does not yet implement the revised independent branch training and
per-branch checkpoint selection. The current mid-fusion runner also retains
legacy all-zero-feature exclusion; common retained track sets need checking
against late fusion before revised comparisons.

The revised matrix keeps the original mixture-only reference and three
nested sets: `M+G+D+P`, `M+G+D+P+B`, and `M+G+D+P+B+V`.
Matched-capacity mixture MLPs, matched mixture ensembles, independent branch
selection, and five-seed aggregation remain pending. Do not treat this
cleanup as completed revised experiments.

## Outputs and next step

Each actual training run writes its checkpoint, resolved config, validation
history, manifest checksum, environment, test metrics, and per-track
predictions into its chosen output directory. No real-data experiments were
run during reorganization.

Next, audit feature provenance and alignment, split IDs, label mapping,
metric definitions, and the actual parameter counts before implementing the
Phase 2 controls. See [the current design](../../../../PHASE2_EXPERIMENTAL_DESIGN.md).

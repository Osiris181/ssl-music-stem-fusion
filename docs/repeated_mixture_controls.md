# Repeated-mixture mid-fusion controls

For `k` source views, compare the stem-fusion concatenation against `k` exact
copies of the same existing mixture embedding. No new audio, separation,
encoder inference, feature normalization, augmentation, or random crop is
introduced by duplication.

| Fusion | Mixture control | Input width for 1024-dimensional features |
| --- | --- | ---: |
| M+G+D+P | M+M+M+M | 4096 |
| M+G+D+P+B | M+M+M+M+M | 5120 |
| M+G+D+P+B+V | M+M+M+M+M+M | 6144 |

Both members of each comparison use the same MLP hidden dimensions (1024/512
in the genre presets), ReLU/dropout, initialization seed, optimizer, learning
rate, batch size, epoch budget, validation checkpoint rule, test metrics,
track ordering, labels, and fixed splits. The implementation counts actual
model parameters and verifies an exact match. Outer seeds remain
32, 64, 128, 256, 512. Unweighted late-fusion mixture ensembles are unchanged:
they are independently trained branches, not repeated inputs to one MLP.

## Interpretation

This is an input-width and nominal parameter-count control, not an independent
information source or proof of equal effective representational capacity.
For identical copies, the first-layer projections can be combined:
`W1*M + ... + Wk*M = (W1 + ... + Wk)*M`. The optimization parameterization
still differs from the ordinary single-mixture MLP. Keep that ordinary
mixture-only reference as well as the repeated controls and avoid calling
copies independent observations or new information.

## Implementation and verified reuse

The shared input-construction helper is `experiments/_mid_controls.py`.
It calls the unchanged `training/trainer.py` backend and includes its own
implementation checksum and repetition count in the model description used
for cache signatures. The standard concatenation fit and independently
trained branch code are unchanged, preserving their existing signatures.
New control names and directories are `repeated_mixture_4`,
`repeated_mixture_5`, and `repeated_mixture_6`. The old `capacity_matched`
condition kind is rejected by the active experiment matrix.

Run `python train_mid.py` from each genre experiment directory to execute the
current seven-condition matrix. Complete compatible ordinary/fusion fits
are verified and reused; only the three new controls need fitting over the
five seeds (15 new fits per dataset, 45 across FMA-Medium, GTZAN, and ISMIR04).
Actual reuse still requires matching data, runtime environment, settings,
paths, implementation identity, and intact artifact checksums. Do not use
`--overwrite` if you intend to reuse the existing fits.

The run rebuilds `summary_mid_fusion.json` and `.txt`, with new paired
differences against the repeated-mixture controls. The readable labels are
M+M+M+M, M+M+M+M+M, and M+M+M+M+M+M. To run a single control/seed, use
`python train_mid.py --run repeated_mixture_4 --seed 32`. A partial command
produces a correspondingly named partial summary, not a complete matrix report.

For future segmented pipelines, select a common mixture segment/window first
and repeat that exact pooled representation. Independently sampled windows
would be a different control, not identical copies.

## Historical archives

The widened-first-hidden-layer controls are superseded. Their original run
artifacts, mid-fusion plans, and combined mid-fusion summaries are preserved
under `outputs/<dataset>/archive/widened_mixture_20261103/` for the three
completed genre datasets. `archive_manifest.json` records every original and
archived path, file size, and checksum. Historical records retain their
original absolute checkpoint paths; use the inventory to locate moved files.
Archive records are not relabeled as repeated-mixture results.

Unchanged fusion runs, the ordinary mixture baseline, late-fusion branches,
late-fusion/stem summaries, features, and ongoing Jamendo extraction stay in
place. Old mid-fusion summaries are not left at the active reporting path
while replacement controls are pending. No new full experiment is started
automatically by this implementation change.

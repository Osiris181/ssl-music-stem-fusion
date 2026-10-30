# Shared training workflows

Dataset entry scripts call these modules; none of the training modules reads
FMA audio paths or hardcodes dataset cache filenames.

| File | Responsibility |
| --- | --- |
| `trainer.py` | Seed initialization, batch loading, model fitting, validation checkpoint selection, test evaluation, artifacts, and verified reuse. |
| `mid_fusion.py` | Concatenate selected embeddings, count actual model parameters, and choose capacity-matched mixture widths. |
| `late_fusion.py` | Train branches independently and assemble equal-logit-sum ensembles from their selected predictions. |
| `metrics.py` | Task loss, validation score, and metric definitions. Genre is implemented first; other task adapters remain future work. |
| `reporting.py` | Atomic artifacts, checksums, mean/sample SD, and paired differences. |

FMA settings live in `configs/training/fma_medium_genre.yaml`. Both fusion
profiles inherit it. The default outer seeds are 32, 64, 128, 256, 512.
Every condition uses the same tracks, Adam at 1e-4, batch size 128, 20 epochs,
ReLU/dropout 0.2, and validation accuracy + macro ROC-AUC + macro average
precision checkpoint selection. Ties retain the earliest epoch.

Run [the FMA entry scripts](../experiments/fma_medium/README.md). Dataset
loading belongs in `data/`; dataset conditions/scheduling belong in
`experiments/`. Additional task adapters can supply different losses,
selection scores, and metrics to the common trainer. Mood/theme and emotion
adapters have not been implemented.

FMA experiment commands automatically reuse only complete runs with matching data, settings,
implementation/software metadata, and artifact checksums. It does not
continue an interrupted optimizer state. `--resume` remains an optional alias
for the default behavior. An incomplete run requires a new
output directory or explicit `--overwrite`. The original trainer is preserved
in `/home/osiris181/stem-fusion-training-archive-20261029/`.

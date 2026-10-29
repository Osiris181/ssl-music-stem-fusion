# Shared training engines

This package consumes cached features and aligned manifests. Public
dataset entry points live in [experiments](../experiments/README.md).

| File | Purpose |
| --- | --- |
| `genre.py` | Single-source or concatenation MLP, cache adapters, metrics, and reproducibility artifacts. |
| `late_fusion.py` | Retained joint-training source MLPs with equal logit summation. |

Use the [FMA-Medium entry points](../experiments/fma_medium/README.md):

```bash
stemfusion train fma-medium mid-fusion --help
stemfusion train fma-medium late-fusion --help
```

Both retain Adam at `1e-4`, batch size 128, 20 epochs, seed 42, hidden
widths 1024/512, dropout 0.2, and validation accuracy + macro ROC-AUC +
macro PR-AUC checkpoint selection.

The public tasks load these defaults from their YAML training presets. Use
`--config` and optional `--run` to select an experiment condition; explicit
CLI arguments override YAML. Effective configs include source-file hashes.

The current late runner selects a whole-model checkpoint after joint training;
it does not yet train/select branches independently. The revised matched
controls and five-seed protocol remain pending. RMS weighting and the old
public wrappers are archived outside the active project.

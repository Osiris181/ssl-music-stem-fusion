# Training configurations

| File | Purpose |
| --- | --- |
| `fma_medium_genre.yaml` | Shared paths, architecture, optimization, metrics, validation selection, and five outer seeds. |
| `fma_medium_mid_fusion.yaml` | Inherits the shared settings and selects concatenation. |
| `fma_medium_late_fusion.yaml` | Inherits the shared settings and selects independent branches with equal logit summation. |

Edit the shared file to change settings consistently for both methods.
The dataset experiment YAMLs in `../experiments/fma_medium/` inherit these
profiles and define the conditions. They are loaded automatically by
`train_mid.py` and `train_late.py`. Paths resolve relative to their declaring
YAML file. Explicit CLI options override YAML and are recorded.

Outer seeds are fixed to `[32, 64, 128, 256, 512]` by default. For a bounded
check, `--seed 32` selects a subset; `--run NAME` selects a condition.
Use a separate output root when changing experimental settings. See
[the training guide](../../docs/fma_htdemucs_training.md).

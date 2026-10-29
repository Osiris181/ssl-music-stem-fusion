# Training configurations

These presets document the retained FMA-Medium classifier settings.

| File | Purpose |
| --- | --- |
| `fma_medium_mid_fusion.yaml` | Single-source and concatenation MLP training. |
| `fma_medium_late_fusion.yaml` | Retained unweighted summed-logit model training. |

`stemfusion train fma-medium mid-fusion` and
`stemfusion train fma-medium late-fusion` automatically load their
corresponding presets. Use
`--config` for experiment files and `--run` for named conditions. See
[configuration usage](../../docs/configuration.md).

Learning rate, optimizer, batch size, epoch budget, hidden widths, dropout,
validation selection, and
seed defaults are unchanged by reorganization.

Source conditions are recorded in `../experiments/fma_medium/`.
Matched controls, independent late-fusion branch selection, and five-seed
reporting still require implementation. Old RMS and duplicate presets are
archived outside the project.

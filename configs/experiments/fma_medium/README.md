# FMA-Medium experiment configurations

| File | Active usage |
| --- | --- |
| `extraction.yaml` | References the feature preset and supplies the fresh MSD/layer-7 output directory. |
| `mid_fusion.yaml` | References the training preset and lists the mixture reference plus three concatenation conditions. |
| `late_fusion.yaml` | References the training preset and lists three equal-sum source-logit conditions. |

Use these files with `--config`. For example:

```bash
stemfusion extract fma-medium --config configs/experiments/fma_medium/extraction.yaml --help
stemfusion train fma-medium mid-fusion --config configs/experiments/fma_medium/mid_fusion.yaml --run mixture_guitar_drums_piano --help
```

For an actual invocation supply the required manifest/local paths.
Extraction uses the YAML `sources` list unless `--source` is provided;
repeat `--source` for a subset. `--help` only displays the interface.
Use `--dry-run` to validate a fully specified configuration without starting
processing. Paths inside YAML are relative to that file; explicit CLI values
take precedence. Without `--run`, supply explicit training sources.

The extraction `sources` list drives the multi-source job in one command.
`other` is excluded from the listed revised conditions.
A training `--run` executes only the selected condition, not the full matrix.

See [configuration usage](../../../docs/configuration.md) and
[the dataset guide](../../../src/stemfusion/experiments/fma_medium/README.md).

These configurations do not implement matched controls, independent branch
selection, or five-seed aggregation. Seed 42 remains the retained default.

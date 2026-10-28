# Paper experiment manifests

Each file in this directory records the runs associated with one results
table. These manifests currently document the run matrix; the command-line
interface does not load YAML files directly.

Current FMA-Medium manifests:

- `table_03_fma_htdemucs.yaml`: HT-Demucs Table 3 mid fusion.
- `table_04_fma_htdemucs.yaml`: HT-Demucs Table 4 classifier comparison.

These are historical matrices, including weighted fusion in Table 4. The
revised primary design is in `../../../PHASE2_EXPERIMENTAL_DESIGN.md`;
its new controls and five-seed configuration files have not been implemented.
Future completed numeric comparisons belong in `reference_results/`.

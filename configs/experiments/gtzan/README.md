# GTZAN experiment configurations

| File | Purpose |
| --- | --- |
| `manifest.yaml` | Provided stratified split files, local audio roots/extensions, explicit unavailable track exclusion, and output manifest. |
| `extraction.yaml` | Shared MusicFM profile, manifest/cache paths, and six selected sources. |
| `mid_fusion.yaml` | Standard mixture baseline, three capacity controls, and three source combinations. |
| `late_fusion.yaml` | Three matched mixture ensembles and three source ensembles. |

Training inherits `configs/training/genre.yaml` through the GTZAN profiles.
FMA-Medium inherits the same common hyperparameters and conventions. Only
class count and dataset paths differ. Conditions and branch seed positions
match the FMA matrices.

Edit local paths in `manifest.yaml` before generating a manifest. Its mixture
root contains genre folders; its HT-Demucs root contains track folders inside
`htdemucs_6s`. Audio validation defaults off to avoid an HDD scan. The explicit
`jazz.00054` exclusion leaves 499/250/250 tracks in the supplied split lists.

Entry scripts automatically load these files. YAML paths resolve relative to
their declaring file; explicit CLI values override them. See the
[GTZAN guide](../../../src/stemfusion/experiments/gtzan/README.md) for commands,
cache compatibility, checkpoint reuse, and summary outputs.

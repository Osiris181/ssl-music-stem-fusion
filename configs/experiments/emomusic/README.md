# EmoMusic preparation, extraction and training presets

| Preset | Purpose |
| --- | --- |
| `feature_manifest.yaml` | Copied Music2Emotion splits/annotations, EmoMusic mixtures and same-ID DEAM stem paths. |
| `separation_manifest.yaml` | Label-independent jobs from the same split IDs and stem destinations; all supplied tracks or an explicit `--track-list` subset. |
| `separation.yaml` | Same shared HT-Demucs profile as DEAM/PMEmo, preserving the intentional DEAM output root and a separate EmoMusic report. |
| `mert.yaml` | Shared pinned MERT profile, retained 30-second tails and mixture-duration alignment. |
| `chords_keys.yaml` | Shared independent BTC/music21 features and bounded CQT retry. |
| `mid_fusion.yaml` | Same seven-condition/five-seed binary model/fitting recipe as DEAM/PMEmo; native scores high >= 5, low < 5, retaining the supplied target columns. |
| `late_fusion.yaml` | Same binary recipe with independent position-paired mixture/stem branches, logit-sum ensembles, protected configured mid references and twelve reports including standalone stems. |

See the [dataset guide](../../../src/stemfusion/experiments/emomusic/README.md)
for commands, outputs and the unresolved annotation-column provenance caveat.
The DEAM stem root is intentional. Paths resolve relative to the
YAML file, and CLI arguments override them. Local metadata is not bundled
public data. Preparation/extraction do not binarize stored scores; the training
preset applies the fixed native midpoint and validation-selected prediction
threshold without rewriting scores or swapping target columns.

`prepare_manifest.py` defaults to the feature purpose; `--purpose separation`
selects the separation preset, as in DEAM/PMEmo. Separation uses the shared
backend with tqdm, complete-stem reuse and a readable summary. Keep
`output_root` as the parent of `htdemucs_6s`, matching the unchanged feature
preset. These additions do not alter existing caches or resolve annotation
provenance; no re-extraction or actual separation is required by the rename.

Late fusion's `reference_output_dir` explicitly selects the mid mixture-only
checkpoint root. Complete compatible references are reused and are never
overwritten by late `--overwrite`. Finish mid training before starting late
training against that root. Existing extraction and mid presets are unchanged.

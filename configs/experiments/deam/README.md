# DEAM preparation and extraction presets

| Preset | Purpose |
| --- | --- |
| `feature_manifest.yaml` | Copied Music2Emotion splits/song-level annotations and existing DEAM mixture/stem roots. |
| `separation_manifest.yaml` | Label-independent jobs from the same split IDs; all supplied tracks by default or an explicit `--track-list` subset. |
| `separation.yaml` | Shared HT-Demucs profile, separation manifest and exact output/report locations. |
| `mert.yaml` | Shared pinned MERT profile, retained 30-second tails and mixture-duration stem alignment. |
| `chords_keys.yaml` | Shared BTC/music21 descriptors, independent per source, with bounded CQT retries. |
| `mid_fusion.yaml` | Same PMEmo model/fitting recipe, seven conditions and five seeds for each target; native 1-9 scores use high >= 5, low < 5 without normalization. |

See the [dataset guide](../../../src/stemfusion/experiments/deam/README.md)
for commands and output locations. Paths are relative to the YAML file, not
the working directory or Music2Emotion checkout. Local metadata is ignored
input that other users must supply. Extraction presets do not binarize scores;
the training preset explicitly retains native scores and uses the fixed
midpoint 5. The probability prediction threshold is selected on validation.

As in PMEmo/Jamendo, `prepare_manifest.py` defaults to the feature preset;
`--purpose separation` selects the separation preset. Both purposes preserve
the supplied split assignments and existing audio destinations. `separation.py`
uses the same shared backend/settings, tqdm progress and readable summary.
Set `output_root` to the parent of `htdemucs_6s`; the model directory must match
`feature_manifest.yaml`'s `htdemucs_root`. Existing feature caches/manifests and
extraction settings do not change; re-extraction is not needed for this layout.

# MTG-Jamendo experiment settings

| File | Purpose |
| --- | --- |
| `separation_manifest.yaml` | Local mood/theme split TSVs, selected-track file, original mixtures, new stem-output root, and separation-manifest destination. |
| `separation.yaml` | Shared HT-Demucs profile, generated manifest, output boundary, and report path. |
| `feature_manifest.yaml` | Complete provided mood/theme splits, existing audio roots, and validated repair mapping for MERT and harmonic extraction. |
| `chords_keys.yaml` | Independent mixture/stem BTC chords and MIDI-derived key paired with MERT, with pinned provenance and verified caches. |
| `mert.yaml` | Pinned frozen MERT, all-layer 30-second embeddings, retained tails, and minimum-only architecture-specific padding. |
| `reproduction.yaml` | Mixture-only author baseline, paper cap of 200 epochs with released validation-loss early stopping (patience 10, delta 0.0001), five seeds, and original harmonic-fusion classifier. |
| `mid_fusion.yaml` | Seven five-seed conditions with shared chord encoding, exact repeated-mixture parameter controls, validated reproduction reuse, and validation-only threshold selection. |
| `late_fusion.yaml` | Independent mixture/stem branches, SHA256 position-paired seeds, matched 4/5/6-branch logit-sum ensembles, and five supporting stem-only rows in the same summary. |

The same separated audio serves MERT and chord/key extraction.
Inference settings come from `../../separators/htdemucs_6s.yaml`.

The flattened `reproduction.yaml` supplies dataset/artifact paths and the
original-mixture input. It inherits
[`mood_theme_tagging.yaml`](../../training/mood_theme_tagging.yaml), then
[`music2emo.yaml`](../../training/music2emo.yaml). The shared recipe retains
the completed reproduction architecture and training/selection rules; its
effective settings and all existing output paths are unchanged. `mert.yaml`
inherits `../../features/mert_v1_95m.yaml` for extraction, independently of
training. `mid_fusion.yaml` inherits the same training recipe and supplies its
condition matrix and independent output root. `late_fusion.yaml` uses the
same recipe with separate branch/ensemble outputs. See the
[mid-fusion guide](../../../src/stemfusion/experiments/mtg_jamendo/README.md#mid-fusion-training)
and [late-fusion guide](../../../src/stemfusion/experiments/mtg_jamendo/README.md#late-fusion-training).

Run `python extract_chords_keys.py` from the dataset-level experiment directory
for the shared harmonic stage. Its output root is
`data/features/mtg_jamendo/harmony/`, separate from the MERT arrays. A move of
the output root alone does not change cached contents or extraction settings.
The report remains at its configured location. See the
[shared chord/key guide](../../../src/stemfusion/experiments/mtg_jamendo/README.md#chordkey-extraction).
`execution.retry_failures: true` enables one automatic pass over supported
tiny-CQT-tail failures after normal extraction. `--no-retry-failures` overrides
it. Verified completed outputs are reused; disabling retries does not invalidate
matching caches. MIDI policy and missing-key behavior are separate settings.

Run `python prepare_manifest.py --purpose separation` then `python separation.py` from the
MTG-Jamendo experiment directory. Both support `--dry-run` and CLI overrides.
Supply selected canonical track IDs in a local text file; the public preset
has no fixed IDs. `--all-tracks` explicitly selects the full mood/theme set.

Run `python prepare_manifest.py` from the same dataset directory to
prepare the shared `mtg_jamendo_features.jsonl` for MERT and chord/key
extraction. It reads `feature_manifest.yaml`, not `separation_manifest.yaml`.
The default purpose is `features`. Both purposes share this entry point but
retain their existing YAML presets and manifest output paths. `--config PATH`
selects another preset for the chosen purpose; explicit CLI values override it.

The separation-manifest and separator presets must specify the same output root because manifest paths
determine exact destinations. Existing stem folders and feature caches are
preserved. See the [dataset guide](../../../src/stemfusion/experiments/mtg_jamendo/README.md).

After generation, `python validate_separation.py --write-wav-fallbacks`
checks decoded waveforms. It reads the same separation preset and writes a
separate `mtg_jamendo_repaired_audio.jsonl` mapping for feature extraction.
Native MP3s remain unchanged; float32 WAV copies address decoder compatibility
where needed. The validation report records the actual decoder and usable path.

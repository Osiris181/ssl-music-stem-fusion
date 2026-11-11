# Music2Emotion chord/key extraction: source comparison

The verified current author release is
[`c5f7984f78d9928b841f25bd05a47377c12c58ca`](https://github.com/AMAAI-Lab/Music2Emotion/commit/c5f7984f78d9928b841f25bd05a47377c12c58ca).
Relevant local author files were compared byte-for-byte with that release;
locally modified extraction scripts were not treated as author code.

## Was MIDI generation missing?

The batch helper
[`preprocess/key_extractor.py`](https://github.com/AMAAI-Lab/Music2Emotion/blob/c5f7984f78d9928b841f25bd05a47377c12c58ca/preprocess/key_extractor.py)
requires existing MIDI files. `preprocess/chordnorm.py` similarly requires
chord and key labels. Neither is an end-to-end MP3 batch command on its own.

However, their complete inference path in
[`music2emo.py`](https://github.com/AMAAI-Lab/Music2Emotion/blob/c5f7984f78d9928b841f25bd05a47377c12c58ca/music2emo.py#L283-L415)
already implements audio → BTC chords → pitch-class MIDI → music21 key →
normalized chords. The first released version,
[`ff69c3f5507ac8522b02c7b97d8bda9da771665d`](https://github.com/AMAAI-Lab/Music2Emotion/blob/ff69c3f5507ac8522b02c7b97d8bda9da771665d/music2emo.py#L254-L389),
also includes MIDI generation. The operations were released, but not joined
into a convenient manifest-driven batch preprocessing command. This MIDI is
a symbolic rendering of chord predictions, not a supplied score or general
polyphonic transcription of the recording.

## Earlier local script

`Music2Emotion/preprocess/extract_jamendo_chords_keys.py` correctly assembles
these stages into a dataset loop. Its BTC normalization, model inference,
interval handling, pitch-class MIDI notes, key analysis, and chord
normalization follow the release. No original MIDI collection is needed.

Practical issues in that helper:

- Its active example input/output paths currently point to EmoMusic.
- Relative model paths depend on launching from `preprocess/`; its repository
  import path is also machine-specific.
- Importing helpers from `music2emo.py` pulls in unrelated UI/model packages.
- It scans all MP3s rather than following the selected mood/theme splits and
  cannot use the prepared manifest's validated WAV alternatives.
- Its existence-only skip logic cannot verify provenance, changed settings,
  corrupted files, or interrupted multi-file writes. Errors are printed but
  there is no full job report.
- Failed key analysis is written as `None major`. The clean command writes
  `None` and records the error, matching the released dataset loader's
  explicit no-key representation; normalization still uses zero shift.
- Both its MIDI loop and the release omit notes for a single-interval chord
  sequence. This is an inherited edge case, not a unique local mistake.

## Clean command

The shared dataset-level entry is
`src/stemfusion/experiments/mtg_jamendo/extract_chords_keys.py`, configured by
`configs/experiments/mtg_jamendo/chords_keys.yaml`. Its harmonic descriptors are
paired with MERT in the active research scope; see the
[usage guide](../src/stemfusion/experiments/mtg_jamendo/README.md#chordkey-extraction).

The new command uses verified BTC utilities and configuration, records
checkpoint hashes/package versions, follows the fixed audio manifest, and
extracts each mixture/stem independently. MIDI is generated and retained by
default. Existing MusicFM features, original audio, and old scripts are not
rewritten. Applying the original mixture pipeline to stems is our extension.

The default preserves the released event policy. The opt-in
`complete_intervals` policy corrects the single-interval MIDI case explicitly.
Key-analysis failures are recorded, not used to drop tracks or fabricate a
major key. Symbolic descriptor encoding and combination with either encoder remain
training concerns; extraction does not invent a flattened MLP representation.

The shared command retains the released CQT frontend in its normal pass.
With default `retry_failures: true`, a single second pass handles recognized
one-sample CQT remainders by right-padding to the validated two-sample minimum.
Supported blocks and ordinary outputs are unchanged; padding and original
sample boundaries are recorded. This is our explicit preprocessing adaptation,
not an author-provided fallback. `--no-retry-failures` disables it. The change
does not fix or reinterpret saved unavailable keys or the single-interval MIDI
case. Successful outputs from the known previous implementation are accepted
only after complete settings, input-identity, and checksum verification.

One training-side compatibility issue remains explicit: BTC's 170-label
vocabulary includes `minmaj7`, but the released 158-entry `chord.json` and
`chord_attr.json` do not encode that quality. The extraction stage preserves
predicted labels; it does not silently discard intervals or map that quality
to a different chord. This must be resolved consistently when implementing
the downstream symbolic-feature loader.

This stage does not reproduce the authors' multitask prediction architecture
or establish identical downstream results. The comparison is MusicFM versus
the MERT+chord/key representation pipeline, not MERT alone.

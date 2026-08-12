# Protocols

Every stage of the pipeline is driven by protocol files rather than directory scans, so
the same utterance list can be reused across augmentation, training, and evaluation.

`build_protocol.py` converts an ASVspoof-style CM protocol into the format used here:

```bash
python protocols/build_protocol.py \
    --input  ASVspoof2019.LA.cm.eval.trl.txt \
    --base   ASVspoof2019_LA_eval/flac \
    --output protocols/ASV19_LA_eval.txt
```

## Formats

Detection protocols — one line per utterance:

```
<file_path> <subset> <label>        # subset ∈ {train, dev, eval}, label ∈ {bonafide, spoof}
eval/augmented/LA_E_9398747__background_music.wav eval spoof
```

Noise classification protocol — the noise class sits between label and subset:

```
<speaker> <file_path> <label> <noise_class> <subset>
LA_0082 .../LA_T_7696598__background_music.wav bonafide background_music train
```

## Protocol sets used in the paper

| Set | Purpose |
|:--|:--|
| `protocol_asv19.txt`, `protocol_df21.txt`, `protocol_itw.txt` | per-benchmark evaluation (Table 2) |
| `protocol_all.txt` | concatenated benchmarks for the pooled EER (Table 2) |
| `noise_type_asv19/`, `noise_type_df21/` | one file per noise class, for per-domain evaluation (Table 4) |
| `catastropy/g0.txt … g7.txt` | one file per domain D0–D7, for the sequential fine-tuning sweep (Figure 2) |
| `NC_protocol.txt` | noise classifier training and evaluation |

The generated protocols are large (hundreds of MB) and reference local audio paths, so
they are not committed; regenerate them with `build_protocol.py` after building the
augmented corpora.

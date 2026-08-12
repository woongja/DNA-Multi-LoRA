# Noise augmentation pipeline

Builds the noise-augmented corpora the whole paper runs on. Utterances from a protocol
are distributed evenly across the noise categories, augmented, and written as
`<utt>__<noise>.wav` — the suffix is the ground-truth noise label used later for
classifier training and for per-domain evaluation.

| File | Role |
|:--|:--|
| `generate_audio.py` | main driver — assignment, parallel augmentation, metadata CSV |
| `generate_audio_autotune.py` | auto-tune runs separately; it is much slower than the other effects |
| `augmentation_config.yaml` | per-effect parameters (SNR, semitones, cutoff frequencies, decay …) |
| `augmentation/` | one module per effect, sharing the `base.py` interface |
| `scripts/generate_{asv19,df21,timit}.sh` | the sharded runs used for the paper, including core allocation |

Effects used for the paper's eight domains: background noise, background music,
auto-tune, band-pass filter, echo, pitch shift, time stretch, Gaussian noise, and
reverberation. `augmentation/` also carries effects explored but not reported (masking,
frequency swap, copy-paste, telephone, amplitude modulation, resampling), which is why it
is larger than the domain list.

Parameter values and the noise sources behind each domain are documented in
[`docs/noise_domains.md`](../docs/noise_domains.md); usage is in
[`docs/reproduce.md`](../docs/reproduce.md).

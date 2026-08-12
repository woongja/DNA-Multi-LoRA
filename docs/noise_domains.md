# Noise domains

The augmented corpora cover ten noise conditions (including clean speech). Similar
conditions are grouped, giving the **eight domains D0–D7** used throughout the paper and
the **nine classes** the noise classification module predicts.

## Domains and augmentation settings (Table 1)

| Domain | Condition | Settings |
|:--|:--|:--|
| D0 | Clean | no augmentation |
| D1 | Background noise | environmental/natural sounds from [ESC-50](https://github.com/karolpiczak/ESC-50), SNR −6 to 3 dB |
| D1 | Background music | [MUSAN](https://www.openslr.org/17/), SNR 10–20 dB |
| D2 | Auto-tune | pitch corrected to the nearest note of a predefined scale ([python-auto-tune](https://github.com/JanWilczek/python-auto-tune)) |
| D3 | Band-pass filter | band-pass (poles = 1) via torchaudio/sox, simulating band-limited speech |
| D4 | Echo | delay and decay sampled at random from predefined ranges |
| D5 | Pitch shift | ±2 to ±5 semitones, `librosa.effects.pitch_shift()` |
| D5 | Time stretch | rate 0.6 and 1.4, `librosa.effects.time_stretch()` (speed change without pitch change) |
| D6 | Gaussian noise | zero-mean, standard deviation sampled in 0.01–0.2 |
| D7 | Reverberation | [RIR](https://www.openslr.org/28/) impulse responses convolved via `fftconvolve` |

SNR ranges follow reported real-world conditions, and pitch shifts are bounded to keep
speech natural and avoid the "chipmunk effect".

## Class ↔ domain mapping

The classifier predicts nine classes; two domains bundle more than one class or
augmentation:

| Classifier class | Domain |
|:--|:--|
| `clean` | D0 |
| `background_noise` | D1 |
| `background_music` | D1 |
| `auto_tune` | D2 |
| `band_pass_filter` | D3 |
| `echo` | D4 |
| `manipulation` | D5 (pitch shift **and** time stretch) |
| `gaussian_noise` | D6 |
| `reverberation` | D7 |

D1 spans two classes because background noise and background music are acoustically
distinct enough to classify apart but are handled by one adapter. D5 is a single class
because pitch shift and time stretch are both speech manipulations that the F0 branch
picks up the same way.

Older augmentation runs emitted `high_pass_filter` and `low_pass_filter` separately;
both map to `band_pass_filter`. `noise_classification/compute_accuracy.py` applies this
mapping when scoring predictions.

## Corpora

Augmentation is applied to LibriSpeech, VCTK, TIMIT, ASVspoof 2019 LA, and ASVspoof 2021
DF, with each corpus split into equal portions so every noise category is evenly
represented. A `*` suffix in the results tables (`ASV19*`, `DF21*`, `TIMIT*`) marks the
noise-augmented version of a corpus.

Detection experiments train on ASV19\* and evaluate on ASV19\*, DF21\*, and In-the-Wild.
The noise classifier trains on LibriSpeech\* and VCTK\* and is evaluated on TIMIT\*, so
the evaluation corpus is unseen and the classifier cannot lean on speaker or linguistic
cues.

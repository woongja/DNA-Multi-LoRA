# Noise classification module

The router's front end: given an utterance, predict which of nine noise conditions it is
in, so the framework can pick the matching LoRA adapter. It has to be accurate enough to
route correctly and small enough to sit in front of a detection model without mattering —
7.2 M parameters, against 87 M for SSAST-base and 319.6 M for a wav2vec 2.0 classifier.

## Model

Audio is cut into 4 s windows and turned into three views, each handled by its own branch:

| View | Branch | What it contributes |
|:--|:--|:--|
| Spectrogram | 3× Conv2d + 2-layer BiLSTM | global spectral distribution — environmental noise, background music |
| MFCC | 3× Conv2d + 2-layer BiLSTM | timbre and formant structure |
| F0 | 2× Conv1d + 2-layer BiLSTM | fundamental-frequency movement — auto-tune, pitch shift |

Each branch mean-pools over frequency, runs the BiLSTM over time, mean-pools over time,
and projects to 1024-d. The three embeddings are concatenated (3072-d) and classified by
a two-layer head with dropout 0.5.

The F0 branch is what makes the module work on *speech manipulations* rather than only
background conditions — real-world audio carries both, and pitch-domain distortions are
invisible to a spectrogram-only classifier.

| File | Role |
|:--|:--|
| `models/multi_feature_fusion_v3.py` | `FusionNet` — the model above |
| `models/NC_Residual.py` | residual variant explored during development |
| `datautils/data_multi_fusion_icassp.py` | dataset computing features on the fly |
| `datautils/data_preprocessed.py` | dataset reading pre-extracted features |
| `preprocess_features.py` | one-off spectrogram/MFCC/F0 extraction |
| `train.py` | training and evaluation entry point |
| `compute_accuracy.py` | scores a prediction file against classes recovered from filenames |
| `scripts/train.sh`, `scripts/eval.sh` | the exact settings used for the paper |
| `notebooks/` | feature pre-processing and evaluation analysis |

## Training

100 epochs, batch 32, Adam at 1e-5, early stopping with patience 3, on 9 classes.
Features are pre-extracted first — recomputing spectrograms every epoch dominates
otherwise. Full commands are in [`docs/reproduce.md`](../docs/reproduce.md).

Training uses LibriSpeech\* and VCTK\*; evaluation uses TIMIT\*. Keeping the evaluation
corpus disjoint forces the classifier onto noise characteristics instead of speaker or
linguistic cues, and it reaches **95 % on TIMIT\*** out of domain.

The class list and how the nine classes map onto the eight paper domains are in
[`docs/noise_domains.md`](../docs/noise_domains.md).

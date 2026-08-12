# DNA Multi LoRA

> **Dynamic Noise-Aware Multi LoRA Framework Towards Real-World Audio Deepfake Detection**
> Woongjae Lee, Hung Dinh-Xuan, Thien-Phuc Doan, Souhwan Jung — Soongsil University, Seoul, Korea
>
> **Accepted at IEEE ICASSP 2026** (Barcelona, Spain, 4–8 May 2026) — oral session SLP-L5,
> *Robust Audio Deepfake and Spoofing Detection I*, pp. 16257–16261.
> [DOI: 10.1109/ICASSP55912.2026.11463424](https://doi.org/10.1109/ICASSP55912.2026.11463424)

Audio deepfake detectors are near-perfect on clean speech and fall apart in the field.
Data augmentation only covers the noise types you thought of in advance, and fine-tuning
on a new noise domain erases what the model learned about the last one.

This repository holds the code and results for **DNA Multi LoRA**, which handles noise by
routing instead of retraining: a lightweight classifier identifies the noise in the input,
and the LoRA adapter trained for that noise domain is attached to a frozen detection
backbone at inference time. Adding a noise domain means training one 27 MB adapter — the
backbone, and every other domain, is left untouched.

**Average EER reduction of 41.4 % over the baselines**, at roughly 1/12 the trainable
parameters of full fine-tuning.

## How it works

```
                     ┌──────────────────────┐
  input audio  ──▶   │ Noise Classification │ ──▶ noise type
                     │  Spec + MFCC + F0    │         │
                     │   CNN-BiLSTM, 7.2M   │         ▼
                     └──────────────────────┘   ┌──────────┐
                                                │  Router  │
                                                └────┬─────┘
                                                     │ selects LoRA_i
                     ┌───────────────────────────────▼──────┐
                     │   frozen SSL backbone (XLSR)         │
                     │   + LoRA_i (rank 8)  →  backend      │ ──▶ real / fake
                     └──────────────────────────────────────┘
```

1. **Classify** — the input is cut into 4 s windows and turned into three views:
   spectrogram (environmental noise), MFCC (timbre/formants), and F0 (pitch
   manipulations such as auto-tune or pitch shift). Three CNN-BiLSTM branches produce
   1024-d embeddings that are concatenated and fused into a 9-way classifier.
2. **Route** — the predicted noise type selects the LoRA adapter trained for that domain.
3. **Detect** — the adapter is merged into the frozen backbone and the utterance is scored.

Nothing about the backbone changes between domains, so there is no catastrophic
forgetting and no full retraining.

## Results

All numbers below are in the tables under [`results/tables/`](results/tables), and every
baseline / fine-tuning figure there is **recomputed from the raw evaluation scores** by
[`results/compute_eer.py`](results/compute_eer.py) rather than copied from the paper.

### Overall EER (%) — [Table 2](results/tables/table2_overall.md)

| Model | Method | Params (M) | ASV19\* | DF21\* | ITW | Pooled |
|:--|:--|--:|--:|--:|--:|--:|
| AASIST-SSL | baseline | — | 15.47 | 21.37 | 10.46 | 21.39 |
| AASIST-SSL | full fine-tuning | 317.8 | 5.03 | 11.35 | 8.05 | 8.13 |
| AASIST-SSL | **DNA Multi LoRA** | **26.7** | 4.65 | 10.85 | 8.88 | **7.93** |
| ConformerTCM | baseline | — | 14.74 | 19.44 | 7.86 | 20.23 |
| ConformerTCM | full fine-tuning | 319.8 | 5.23 | 11.08 | 8.14 | 8.14 |
| ConformerTCM | **DNA Multi LoRA** | **27.2** | 4.37 | 10.30 | 8.13 | **7.55** |

\* our noise-augmented versions of ASVspoof 2019 LA and ASVspoof 2021 DF. ITW is
[In-the-Wild](https://arxiv.org/abs/2203.16263), used as-is.

### Per-domain EER (%) on ASV19\* — [Table 4](results/tables/table4_per_domain.md)

| Model | Method | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| AASIST-SSL | baseline | 0.15 | 3.67 | 3.86 | 15.37 | 10.42 | 22.87 | 8.75 | 40.01 |
| AASIST-SSL | noise-specific LoRA | — | 2.30 | 0.70 | 1.79 | 0.92 | 4.19 | 3.57 | 4.38 |
| ConformerTCM | baseline | 0.15 | 4.02 | 11.00 | 9.59 | 24.11 | 20.60 | 5.49 | 29.56 |
| ConformerTCM | noise-specific LoRA | — | 1.42 | 0.87 | 1.94 | 1.18 | 4.57 | 3.26 | 3.40 |

The hardest domains for the baselines are exactly the ones a single adapter fixes:
reverberation (D7) drops 40.01 → 4.38, echo (D4) 24.11 → 1.18. Domain definitions are in
[`docs/noise_domains.md`](docs/noise_domains.md).

### Catastrophic forgetting — [Figure 2](results/tables/fig2_sequential_ft.md)

Fine-tuning sequentially on D1 → D7 and re-evaluating earlier domains: ConformerTCM
degrades from 0.17 to 5.04 EER, AASIST-SSL from 0.03 to 4.81. The CSV holds the full
grid (every *fine-tuned-on* × *evaluated-on* pair), not just the plotted curve.

### Noise classifier — Table 3

| Model | Features | Params | Accuracy (TIMIT\*, out-of-domain) |
|:--|:--|--:|--:|
| Wav2Vec 2.0 + Linear | representation vector | 319.6 M | 91 % |
| SSAST-base | spectrogram | 87 M | 98.5 % |
| **CNN-BiLSTM (ours)** | **MFCC + F0 + spectrogram** | **7.2 M** | **95 %** |

Trained on LibriSpeech\* and VCTK\*, evaluated on TIMIT\* so the classifier has to key on
noise characteristics rather than speaker or linguistic content. Re-score a prediction
file with [`noise_classification/compute_accuracy.py`](noise_classification/compute_accuracy.py).

## Repository layout

| Path | What is in it |
|:--|:--|
| `noise_classification/` | The noise classification module — three-branch CNN-BiLSTM fusion model, dataloaders, feature pre-extraction, training and evaluation entry points |
| `noise_augmentation/` | The augmentation pipeline that builds the noise-augmented corpora (Table 1): background noise/music, auto-tune, band-pass, echo, pitch shift, time stretch, Gaussian noise, reverberation |
| `lora/` | LoRA configuration — the per-domain adapter configs, dynamic-routing configs, and the run scripts for every ICASSP26 experiment |
| `detection/` | Our modifications to the two detection backbones, as applyable patches plus the changed files |
| `protocols/` | Protocol construction for the augmented corpora |
| `results/` | EER computation, generated tables, figures, and analysis notebooks |
| `docs/` | Noise domain definitions and the full reproduction walkthrough |

## Getting started

```bash
git clone https://github.com/woongja/DNA-Multi-LoRA.git
cd DNA-Multi-LoRA
pip install -r requirements.txt
```

Then follow [`docs/reproduce.md`](docs/reproduce.md), which walks through building the
augmented corpora, training the noise classifier, training one LoRA adapter per domain,
and running the dynamic-routing evaluation.

### What is not in this repository

Model weights are not distributed here. The XLSR-based backbones are ~1.3 GB each and
belong to their upstream projects; get them from
[SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing) (AASIST-SSL) and
[tcm_add](https://github.com/ductuantruong/tcm_add) (ConformerTCM). The trained noise
classifier and the LoRA adapters are also kept out of version control — see
`docs/reproduce.md` for how to retrain them.

Raw evaluation scores (~1.3 GB across 146 files) are likewise excluded; the derived
tables in `results/tables/` are committed instead, and `results/compute_eer.py`
regenerates them from a local copy of the scores.

## Built on

- [SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing) — AASIST-SSL detection backbone
- [tcm_add](https://github.com/ductuantruong/tcm_add) — ConformerTCM detection backbone
- [PEFT](https://github.com/huggingface/peft) — LoRA implementation
- Datasets: ASVspoof 2019 LA, ASVspoof 2021 DF, In-the-Wild, LibriSpeech, VCTK, TIMIT
- Noise sources: ESC-50, MUSAN, RIR

## Citation

```bibtex
@inproceedings{lee2026dynamic,
  title={Dynamic Noise-Aware Multi Lora Framework Towards Real-World Audio Deepfake Detection},
  author={Lee, Woongjae and Dinh-Xuan, Hung and Doan, Thien-Phuc and Jung, Souhwan},
  booktitle={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={16257--16261},
  year={2026},
  organization={IEEE}
}
```

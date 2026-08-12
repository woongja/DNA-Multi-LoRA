# Detection backbones

Both detectors are upstream projects that we modified rather than reimplemented, so this
directory ships our changes, not a fork.

| Directory | Upstream | Model |
|:--|:--|:--|
| `aasist_ssl/` | [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing) | AASIST-SSL (wav2vec 2.0 + AASIST) |
| `conformertcm/` | [ductuantruong/tcm_add](https://github.com/ductuantruong/tcm_add) | ConformerTCM (temporal-channel modelling in MHSA) |

Each directory holds:

- **`upstream_modifications.patch`** — our diff against the upstream commit. Clone the
  upstream repository and `git apply` it.
- **`modified/`** — the same changed files as plain sources, for reading without applying
  the patch, plus the `*_embedding.py` variants that dump penultimate-layer embeddings for
  the t-SNE analysis in Figure 3.
- **`scripts/`** — the training and evaluation runs used in the paper.

```bash
git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git
cd SSL_Anti-spoofing
git apply ../detection/aasist_ssl/upstream_modifications.patch
```

## What the scripts do

| Script | Purpose |
|:--|:--|
| `train.sh` | baseline training on the clean protocol |
| `train_seperate.sh` | full fine-tuning per noise domain — the `finetune` row of Table 2 |
| `eval.sh`, `eval_2.sh` | evaluation on ASV19\*, DF21\*, In-the-Wild |
| `eval_noise.sh` | per-noise-domain evaluation — the `baseline` row of Table 4 |
| `eval_cata.sh` | sequential fine-tuning sweep D1 → D7 behind Figure 2 |
| `tsne.sh` | embedding extraction for Figure 3 |

Paths in the scripts are placeholders (`${PROJECT_ROOT}`, `${PREPROCESS_DIR}`); set them
for your machine first. Pre-trained backbone weights (~1.3 GB each) come from the
upstream repositories and are not distributed here.

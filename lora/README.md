# LoRA configuration

The configuration behind every adapter in the paper: one experiment config per noise
domain, the dynamic-routing configs, and the run scripts that produced the reported
numbers. Adapter training itself runs in a PyTorch Lightning + Hydra training harness,
so these are the files that define *what* was trained and evaluated, not the trainer.

## Contents

| Path | Role |
|:--|:--|
| `configs/icassp26/{aasist_ssl,conformertcm}/*_single_lora.yaml` | one fixed adapter — used for the per-domain results in Table 4 |
| `configs/icassp26/{aasist_ssl,conformertcm}/*_mul_lora.yaml` | dynamic routing — the noise classifier selects the adapter per utterance |
| `configs/icassp26_lora_loss_earlystop_logger.yaml` | callback set for adapter training (loss monitoring, early stopping, logging) |
| `scripts/{aasist_ssl,conformertcm}_g{1..7}.sh` | one adapter per noise domain, D1 → D7 |
| `scripts/{aasist_ssl,conformertcm}_dynamic_lora.sh` | dynamic-routing runs |
| `scripts/benchmark.sh`, `scripts/benchmark_var.sh` | evaluation drivers — fixed adapter and dynamic routing respectively |
| `scripts/run_all_benchmarks{,_df}.sh` | sweep every model × domain combination |
| `scripts/README.md` | the exact command and adapter checkpoint behind each reported number |
| `src/` | the routing module that reads the noise classifier's prediction and attaches the matching adapter to the frozen backbone |
| `.env.example` | dataset roots, backbone checkpoints, adapter list, optional tracker tokens |

## Setup

Copy `configs/icassp26` into the harness's `configs/experiment/`, the callback config into
`configs/callbacks/`, and the scripts into `scripts/icassp26/`. Fill in `.env.example` as
`.env` — it is git-ignored on purpose, so **never commit a filled-in copy**.

## Adapter configuration

Each adapter is trained independently on ASV19\* for one noise domain, at **rank 8**, with
the backbone frozen. A trained adapter is a standard PEFT directory
(`adapter_model.safetensors` + `adapter_config.json`, ~14 MB), so adding a noise domain
costs one adapter and leaves every other domain untouched — this is what avoids the
catastrophic forgetting that sequential fine-tuning shows in Figure 2.

Rank 8 was chosen over the rank 4 reported as a good efficiency/performance trade-off for
wav2vec 2.0-based detection: in noisy domains rank 8 was consistently better while still
being parameter-efficient.

At inference the router reads the noise classifier's prediction and loads the
corresponding adapter; `benchmark_var.sh` is the entry point for that path.

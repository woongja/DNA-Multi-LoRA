# Noise-specific LoRA and dynamic routing

Adapter training and the routing evaluation run inside
[hungdinhxuan/Lightning-hydra](https://github.com/hungdinhxuan/Lightning-hydra), a
PyTorch Lightning + Hydra framework for anti-spoofing experiments. This directory is the
ICASSP26 slice of that tree — the files you need to drop into a clone of it.

## Contents

| Path | Role |
|:--|:--|
| `src/models/base/normal_nc_dynamic_lora_module.py` | The dynamic framework: runs the noise classifier, picks the matching adapter, attaches it to the frozen backbone, and scores |
| `src/models/components/noise_classifier.py` | Noise classifier wrapper used by the router at inference time |
| `src/models/xlsr_conformertcm_baseline_mdt_lora_module.py` | ConformerTCM + LoRA training module |
| `src/callbacks/lora_model_checkpoint_callback.py` | Saves adapter weights only, not the frozen backbone |
| `configs/icassp26/` | Experiment configs — `*_single_lora.yaml` (one fixed adapter) and `*_mul_lora.yaml` (dynamic routing) for both backbones |
| `configs/icassp26_lora_loss_earlystop_logger.yaml` | Callback set used for adapter training |
| `scripts/{aasist_ssl,conformertcm}_g{1..7}.sh` | One adapter per noise domain, D1 → D7 |
| `scripts/{aasist_ssl,conformertcm}_dynamic_lora.sh` | Dynamic-routing runs |
| `scripts/benchmark.sh`, `scripts/benchmark_var.sh` | Evaluation drivers — fixed adapter and dynamic routing respectively |
| `scripts/run_all_benchmarks{,_df}.sh` | Sweep every model/domain combination |
| `scripts/README.md` | The exact commands and adapter checkpoints behind each table |

## Setup

```bash
git clone https://github.com/hungdinhxuan/Lightning-hydra.git
cd Lightning-hydra
cp -r <this-repo>/lora/src/* src/
cp -r <this-repo>/lora/configs/icassp26 configs/experiment/
cp <this-repo>/lora/configs/icassp26_lora_loss_earlystop_logger.yaml configs/callbacks/
cp -r <this-repo>/lora/scripts/* scripts/icassp26/
cp <this-repo>/lora/.env.example .env   # then fill it in
```

`.env` carries dataset roots, backbone checkpoint paths, the adapter list, and optional
experiment-tracker tokens. **It is git-ignored on purpose — do not commit a filled-in
copy.**

## Adapter configuration

Each adapter is trained independently on ASV19\* for one noise domain, at **rank 8**,
with the backbone frozen. A trained adapter is a standard PEFT directory
(`adapter_model.safetensors` + `adapter_config.json`, ~14 MB), so adding a noise domain
costs one adapter and leaves every other domain untouched — this is what avoids the
catastrophic forgetting that sequential fine-tuning shows in Figure 2.

At inference the router reads the noise classifier's prediction and loads the
corresponding adapter; `benchmark_var.sh` is the entry point for that path.

# Reproducing the experiments

The pipeline has four stages: build the noise-augmented corpora, train the noise
classification module, train one LoRA adapter per noise domain, then evaluate with
dynamic routing. Stages 3 and 4 need the detection backbones from their upstream
projects.

Paths in the committed scripts are placeholders (`${PROJECT_ROOT}`, `${PREPROCESS_DIR}`,
`${ADAPTER_LOG_ROOT}`, …). Set them for your machine before running anything.

## 0. Prerequisites

| Item | Where to get it |
|:--|:--|
| ASVspoof 2019 LA, ASVspoof 2021 DF | [ASVspoof](https://www.asvspoof.org/) |
| In-the-Wild | [Müller et al., 2022](https://deepfake-total.com/in_the_wild) |
| LibriSpeech, VCTK, TIMIT | OpenSLR / CSTR / LDC |
| MUSAN (background music), ESC-50 (environmental noise), RIR | [OpenSLR 17](https://www.openslr.org/17/), [ESC-50](https://github.com/karolpiczak/ESC-50), [OpenSLR 28](https://www.openslr.org/28/) |
| XLSR-2B / wav2vec 2.0 front end | [fairseq](https://github.com/facebookresearch/fairseq) |
| AASIST-SSL backbone | [SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing) |
| ConformerTCM backbone | [tcm_add](https://github.com/ductuantruong/tcm_add) |

```bash
pip install -r requirements.txt
```

## 1. Build the noise-augmented corpora

Protocols first — one line per utterance, `<path> <subset> <label>`:

```bash
python protocols/build_protocol.py \
    --input  <asvspoof_cm_protocol>.txt \
    --base   <flac_directory> \
    --output protocols/ASV19_LA_eval.txt
```

Then augment. `generate_audio.py` distributes utterances evenly across the noise
categories, writes the augmented audio, and records a metadata CSV; auto-tune runs in a
separate process because it is far slower than the rest:

```bash
python noise_augmentation/generate_audio.py \
    --protocol protocols/ASV19_LA_{train,dev,eval}.txt \
    --out-root  $OUT_ROOT \
    --aug-config noise_augmentation/augmentation_config.yaml \
    --copy-clean --seed 20250909 --sr 16000

python noise_augmentation/generate_audio_autotune.py \
    --protocol protocols/ASV19_LA_{train,dev,eval}.txt \
    --out-root $OUT_ROOT --aug-config noise_augmentation/augmentation_config.yaml
```

`noise_augmentation/scripts/generate_{asv19,df21,timit}.sh` wrap both calls with the core
allocation and sharding actually used for the paper. Augmented files are named
`<utt>__<noise>.wav`, which is how the ground-truth noise class is recovered later.

Individual effects live in `noise_augmentation/augmentation/` (one module per effect,
parameters in `augmentation_config.yaml`); see
[`docs/noise_domains.md`](noise_domains.md) for the settings used per domain.

## 2. Train the noise classification module

Features are pre-extracted once — spectrogram, MFCC, and F0 per 4 s window — because
recomputing them every epoch dominates training time:

```bash
python noise_classification/preprocess_features.py \
    --protocol_file protocols/NC_protocol.txt \
    --output_dir    $PREPROCESSED_DIR \
    --subset all
```

Then train the three-branch fusion model (9 classes, see the class↔domain mapping in
`docs/noise_domains.md`):

```bash
python noise_classification/train.py --is_train \
    --preprocessed_dir $PREPROCESSED_DIR \
    --batch_size 32 --num_epochs 100 --learning_rate 1e-5 \
    --num_classes 9 --early_stop_patience 3 \
    --input_height 128 --input_width 126 --f0_len 126 \
    --save_path out/icassp_v3.pth --log_dir runs/icassp_preprocessed
```

Evaluate on the held-out corpus and score the predictions:

```bash
python noise_classification/train.py --is_eval \
    --preprocessed_dir $PREPROCESSED_DIR \
    --model_path out/icassp_v3.pth --num_classes 9 \
    --input_height 128 --input_width 126 --f0_len 126 \
    --save_results results/NC_timit.txt

python noise_classification/compute_accuracy.py \
    --predictions results/NC_timit.txt --confusion
```

`noise_classification/scripts/{train,eval}.sh` hold the exact settings; the training
script pre-processes automatically if `$PREPROCESSED_DIR` does not exist yet.

## 3. Train the noise-specific LoRA adapters

Adapter training runs inside the
[Lightning-hydra](https://github.com/hungdinhxuan/Lightning-hydra) framework. Clone it,
copy `lora/src/**` over the matching paths, drop `lora/configs/icassp26` into
`configs/experiment/`, and fill in `lora/.env.example` as `.env` (dataset roots, backbone
checkpoints, output directory — **never commit the filled-in file**).

One adapter per domain, rank 8, trained on ASV19\* while the backbone stays frozen:

```bash
# from the Lightning-hydra root
bash scripts/icassp26/aasist_ssl_g1.sh        # D1 background music/noise
bash scripts/icassp26/aasist_ssl_g2.sh        # D2 auto-tune
...                                           # g3 … g7
bash scripts/icassp26/conformertcm_g1.sh      # same seven domains, ConformerTCM
```

Each run writes a PEFT adapter (`adapter_model.safetensors` + `adapter_config.json`,
~14 MB). `lora/scripts/README.md` lists the checkpoint each experiment in the paper used.

Rank 8 was chosen over the rank 4 reported as a good efficiency/performance trade-off for
wav2vec 2.0-based detection: in noisy domains rank 8 was consistently better while still
being parameter-efficient.

## 4. Evaluate with dynamic routing

`lora/scripts/benchmark_var.sh` is the dynamic-routing evaluation — the noise classifier
picks the adapter per utterance. `benchmark.sh` runs a single fixed adapter, which is how
the per-domain numbers in Table 4 were produced:

```bash
# dynamic routing (DNA Multi LoRA)
./scripts/benchmark_var.sh -g 2 -c icassp26/aasist_ssl/xlsr_aasist_mul_lora \
    -b data/ICASSP25_benchmark_noise -m $BASE_MODEL -r $RESULTS_DIR -n dynamic_lora

# one fixed adapter
./scripts/benchmark.sh -g 2 -c icassp26/aasist_ssl/xlsr_aasist_single_lora \
    -b data/ICASSP25_benchmark_noise -m $BASE_MODEL -a $ADAPTER_CKPT -r $RESULTS_DIR -n single_lora
```

Baseline and full fine-tuning references come from the detection repositories directly —
see `detection/{aasist_ssl,conformertcm}/`, where `upstream_modifications.patch` applies
our changes to a fresh upstream clone and `scripts/` holds the training and evaluation
runs (`train_seperate.sh` for per-domain fine-tuning, `eval_cata.sh` for the sequential
fine-tuning sweep behind Figure 2).

## 5. Recompute the tables

Every evaluation writes a score file (`<utt_id> <score>`, or
`<utt_id> <spoof> <bonafide>` for AASIST-SSL). Point the script at your score and
protocol roots:

```bash
python results/compute_eer.py \
    --protocols <protocol_root> \
    --scores    <score_root> \
    --out       results/tables
```

This regenerates `table2_overall`, `table4_per_domain`, and `fig2_sequential_ft` as both
CSV and Markdown. The `dna_multi_lora` rows are quoted from the paper — those score files
were produced inside the Lightning-hydra tree and are not part of this snapshot; every
other number is computed from the raw scores.

`results/notebooks/` holds the analysis behind the figures: `evaluation_eer.ipynb`
(per-pair EER and confusion matrices), `graph.ipynb` (Figure 2), and
`tsne_conformertcm_baseline_clean.ipynb` (Figure 3, the t-SNE comparison of baseline,
fine-tuned, and D5-specific LoRA representations).

#!/usr/bin/env bash
set -eo pipefail

# ===== 경로 설정 =====
PROTO_DIR="${PROJECT_ROOT}/protocols/noise_type_df21"
MODEL_PATH="${PROJECT_ROOT}/tcm_add/avg_5_best.pth"
DB_PATH="${PROJECT_ROOT}/Dataset/icassp/df21"
OUT_DIR="${PROJECT_ROOT}/results/noise"
LOG_DIR="${OUT_DIR}/logs_tcm"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# ===== GPU 설정 =====
GPUS=(1 2)     # 사용할 GPU 번호들
PER_GPU_JOBS=1   # GPU당 동시에 몇 개까지 돌릴지

GPU_COUNT=${#GPUS[@]}
MAX_CONCURRENCY=$(( GPU_COUNT * PER_GPU_JOBS ))

echo "[INFO] Running ConformerTCM baseline"
echo "[INFO] GPUs: ${GPUS[*]} (max_concurrency=${MAX_CONCURRENCY})"
echo

i=0
shopt -s nullglob
for proto in "${PROTO_DIR}"/*.txt; do
  name="$(basename "$proto" .txt)"   # ex) auto_tune, clean, reverberation
  while [ "$(jobs -r | wc -l)" -ge "$MAX_CONCURRENCY" ]; do sleep 1; done

  gpu="${GPUS[$(( i % GPU_COUNT ))]}"; i=$((i+1))
  out_file="${OUT_DIR}/conformertcm-df21-${name}.txt"
  log_file="${LOG_DIR}/${name}.log"

  echo "[RUN] ${name} -> GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python main.py \
    --protocols_path "${proto}" \
    --database_path "${DB_PATH}" \
    --average_model false \
    --model_path "${MODEL_PATH}" \
    --eval_output "${out_file}" \
    > "${log_file}" 2>&1 &
done

wait
echo "[DONE] All TCM runs finished."

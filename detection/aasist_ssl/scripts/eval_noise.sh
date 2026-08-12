#!/usr/bin/env bash
set -euo pipefail

# ====== 경로/설정 ======
PROTO_DIR="${PROJECT_ROOT}/protocols/noise_type_df21"
MODEL_PATH="${PROJECT_ROOT}/SSL_Anti-spoofing/Best_LA_model_for_DF.pth"
DB_PATH="${PROJECT_ROOT}/Dataset/icassp/df21"
OUT_DIR="${PROJECT_ROOT}/results/noise"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# 사용할 GPU 목록과 1개 GPU당 동시 실행 개수
GPUS=(1 2)          # 예: 0,1,2번 카드 사용
PER_GPU_JOBS=1        # GPU당 동시에 몇 개까지 돌릴지 (원하면 2로 올려도 됨)

# ====== 내부 파라미터 ======
GPU_COUNT=${#GPUS[@]}
MAX_CONCURRENCY=$(( GPU_COUNT * PER_GPU_JOBS ))

echo "[INFO] Protocol dir : $PROTO_DIR"
echo "[INFO] Output dir    : $OUT_DIR"
echo "[INFO] GPUs          : ${GPUS[*]} (per-GPU=$PER_GPU_JOBS, max_concurrency=$MAX_CONCURRENCY)"
echo

i=0
shopt -s nullglob
for proto in "${PROTO_DIR}"/*.txt; do
  name="$(basename "$proto" .txt)"

  # 동시 실행 상한에 도달하면 슬롯 날 때까지 대기
  while [ "$(jobs -r | wc -l)" -ge "$MAX_CONCURRENCY" ]; do
    sleep 1
  done

  # 라운드로빈으로 GPU 배정
  gpu="${GPUS[$(( i % GPU_COUNT ))]}"
  i=$(( i + 1 ))

  out_file="${OUT_DIR}/aasist-df21-${name}.txt"
  log_file="${LOG_DIR}/${name}.log"

  echo "[RUN] ${name} -> GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python main_SSL_DF.py \
    --track=DF \
    --is_eval \
    --eval \
    --model_path "${MODEL_PATH}" \
    --protocols_path "${proto}" \
    --database_path "${DB_PATH}" \
    --eval_output "${out_file}" \
    > "${log_file}" 2>&1 &

done

# 남은 작업 모두 기다림
wait
echo "[DONE] All protocols finished."

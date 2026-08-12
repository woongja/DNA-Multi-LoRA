#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
# -------------------------
# 공통 설정
# -------------------------
source ~/miniconda3/etc/profile.d/conda.sh

TOTAL_CORES=$(nproc)
GEN_CORES=$(( TOTAL_CORES*7/10 ))
AT_CORES=$(( TOTAL_CORES - GEN_CORES ))
GEN_CORES=$(( GEN_CORES>0 ? GEN_CORES : 1 ))
AT_CORES=$(( AT_CORES>0 ? AT_CORES : 1 ))

echo "[INFO] Cores: total=$TOTAL_CORES → general=$GEN_CORES, autotune=$AT_CORES"

# 경로들(네가 준 경로들 유지)
OUT_ROOT="${PREPROCESS_DIR}/icassp/asv19"
AUG_CFG="${PROJECT_ROOT}/Dataset/NoiseDataset/augmentation_config.yaml"
ASSIGN_DIR="${OUT_ROOT}/_assigned"
LOG_DIR="${OUT_ROOT}/_logs"
mkdir -p "$ASSIGN_DIR" "$LOG_DIR"

GEN_SCRIPT="${PROJECT_ROOT}/Dataset/NoiseDataset/generate_audio.py"
AT_SCRIPT="${PROJECT_ROOT}/Dataset/NoiseDataset/generate_audio_autotune.py"

PROTO_DIR="${PROJECT_ROOT}/protocols"
# 분배는 하나의 통합 프로토콜로 해도 되고, 여러 개를 합쳐도 됨.
# 여기서는 train/dev/eval 전부 넣어서 전체에서 균등 분배하도록 함.
PROTO_TRAIN="${PROTO_DIR}/ASV19_LA_train.txt"
PROTO_DEV="${PROTO_DIR}/ASV19_LA_dev.txt"
PROTO_EVAL="${PROTO_DIR}/ASV19_LA_eval.txt"

SEED="20250909"
SR=16000

ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "======================================"
echo "$(ts) [STEP 0] 분배(샤드 생성) 시작"
echo "→ ${ASSIGN_DIR}"
echo "======================================"

# -------------------------
# STEP 0: 분배(샤드 생성)
# -------------------------
conda run -n dataset --no-capture-output \
  python "$GEN_SCRIPT" \
    --protocol "$PROTO_TRAIN" "$PROTO_DEV" "$PROTO_EVAL" \
    --out-root "$OUT_ROOT" \
    --aug-config "$AUG_CFG" \
    --dump-assignments-dir "$ASSIGN_DIR" \
    --include-auto-tune \
    --include-clean \
    --seed "$SEED" \
  |& tee -a "$LOG_DIR/step0_assign_$(date +%Y%m%d_%H%M%S).log"

echo "$(ts) [STEP 0 DONE] 분배 파일 목록:"
ls -1 "$ASSIGN_DIR" | sed 's/^/ - /'

# -------------------------
# STEP 1: 일반 증강 (병렬 실행 준비)
# -------------------------
run_general_noise() {
  echo "$(ts) [STEP 1] 일반 증강 시작"

  local SUBSETS=(train dev eval)
  local AUGS=(background_noise background_music gaussian_noise high_pass_filter low_pass_filter pitch_shift time_stretch echo reverberation)

  for SUB in "${SUBSETS[@]}"; do
    local SUB_DIR="${ASSIGN_DIR}/${SUB}"
    [[ -d "$SUB_DIR" ]] || { echo "[INFO] skip ${SUB} (no shards)"; continue; }

    echo "$(ts)  -> subset=${SUB}"
    mapfile -t SHARDS < <(find "$SUB_DIR" -maxdepth 1 -type f -name "*.txt" ! -name "auto_tune.txt" | sort)
    if [[ ${#SHARDS[@]} -eq 0 ]]; then
      echo "[INFO]   no general shards in ${SUB}"
      continue
    fi

    for F in "${SHARDS[@]}"; do
      AUG_NAME="$(basename "${F%.txt}")"
      LINES=$(wc -l < "$F" || echo 0)
      echo "$(ts)    - ${SUB}/${AUG_NAME}: ${LINES} lines"

      if [[ "$AUG_NAME" == "clean" ]]; then
        # CLEAN 전용 실행
        EXCLUDE_LIST=( "${AUGS[@]}" )  # 모든 증강 제외
        conda run -n dataset --no-capture-output \
          python "$GEN_SCRIPT" \
            --protocol "$F" \
            --out-root "$OUT_ROOT" \
            --aug-config "$AUG_CFG" \
            --sr "$SR" \
            --copy-clean \
            --skip-existing \
            --force-subset "$SUB" \
            --num-workers "$GEN_CORES" \
            --meta-out meta_noise.csv \
            --exclude "${EXCLUDE_LIST[@]}" \
            --clean-only \
        |& tee -a "$LOG_DIR/step1_${SUB}_clean_$(date +%Y%m%d_%H%M%S).log"
      else
        # 특정 증강만 실행 (clean 메타는 만들지 않음)
        EXCLUDE_LIST=()
        for A in "${AUGS[@]}"; do
          [[ "$A" == "$AUG_NAME" ]] || EXCLUDE_LIST+=( "$A" )
        done
        conda run -n dataset --no-capture-output \
          python "$GEN_SCRIPT" \
            --protocol "$F" \
            --out-root "$OUT_ROOT" \
            --aug-config "$AUG_CFG" \
            --sr "$SR" \
            --copy-clean \
            --skip-existing \
            --force-subset "$SUB" \
            --num-workers "$GEN_CORES" \
            --meta-out meta_noise.csv \
            --exclude "${EXCLUDE_LIST[@]}" \
            --skip-clean-meta \
        |& tee -a "$LOG_DIR/step1_${SUB}_${AUG_NAME}_$(date +%Y%m%d_%H%M%S).log"
      fi
    done
  done

  echo "$(ts) [STEP 1 DONE] 일반 증강 완료"
}

# -------------------------
# STEP 2: 오토튠 (병렬 실행 준비)
# -------------------------
run_autotune() {
  echo "$(ts) [STEP 2] 오토튠 시작"
  local SUBSETS=(train dev eval)

  for SUB in "${SUBSETS[@]}"; do
    local AT_PROTO="${ASSIGN_DIR}/${SUB}/auto_tune.txt"
    if [[ ! -f "$AT_PROTO" ]]; then
      echo "[INFO]  skip ${SUB}/auto_tune (no shard)"
      continue
    fi
    LINES=$(wc -l < "$AT_PROTO" || echo 0)
    echo "$(ts)   - ${SUB}/auto_tune: ${LINES} lines"
    conda run -n auto-tune --no-capture-output \
      python "$AT_SCRIPT" \
        --protocol "$AT_PROTO" \
        --out-root "$OUT_ROOT" \
        --aug-config "$AUG_CFG" \
        --sr "$SR" \
        --copy-clean \
        --skip-existing \
        --force-subset "$SUB" \
        --meta-out meta_noise_autotune.csv \
    |& tee -a "$LOG_DIR/step2_${SUB}_autotune_$(date +%Y%m%d_%H%M%S).log"
  done

  echo "$(ts) [STEP 2 DONE] 오토튠 완료"
}

# -------------------------
# 병렬 실행
# -------------------------
echo "======================================"
echo "$(ts) 병렬 실행 시작 (STEP 1 & STEP 2)"
echo "======================================"

run_general_noise &
PID1=$!

run_autotune &
PID2=$!

echo "$(ts) 두 작업 종료 대기…"
wait $PID1; EXIT1=$?
wait $PID2; EXIT2=$?

echo "======================================"
if [[ $EXIT1 -eq 0 && $EXIT2 -eq 0 ]]; then
  echo "$(ts) [ALL DONE] 일반+오토튠 성공"

  # ✅ 메타 병합 (변수 치환 되도록 <<PY 사용: 따옴표 없음)
  conda run -n dataset --no-capture-output python - <<PY
import os, pandas as pd

root = os.path.expandvars("$OUT_ROOT")

pairs = [
    ("meta_noise.csv",              "meta_noise_autotune.csv",              "meta_noise_all.csv"),
    ("meta_noise_df21.csv",         "meta_noise_autotune_df21.csv",         "meta_noise_all_df21.csv"),
]

made_any = False
for a, b, out in pairs:
    A = os.path.join(root, a)
    B = os.path.join(root, b)
    if not (os.path.exists(A) or os.path.exists(B)):
        continue
    A_df = pd.read_csv(A) if os.path.exists(A) else pd.DataFrame()
    B_df = pd.read_csv(B) if os.path.exists(B) else pd.DataFrame()
    df = pd.concat([A_df, B_df], ignore_index=True)
    if "file_path" in df.columns:
        df = df.drop_duplicates(subset=["file_path"])
    outp = os.path.join(root, out)
    df.to_csv(outp, index=False)
    print(f"[INFO] merged meta -> {outp} ({len(df)} rows)")
    made_any = True

if not made_any:
    print("[WARN] No meta files found to merge under", root)
PY

  echo "Logs:"
  ls -1 "$LOG_DIR"
  echo "======================================"
  exit 0
else
  echo "$(ts) [ERROR] 일부 실패"
  echo "General noise exit code: $EXIT1"
  echo "Auto-tune exit code:    $EXIT2"
  echo "---- Last 50 lines (general) ----"
  tail -n 50 "$LOG_DIR"/step1_general_* 2>/dev/null || true
  echo "---- Last 50 lines (autotune) ----"
  tail -n 50 "$LOG_DIR"/step2_autotune_* 2>/dev/null || true
  echo "======================================"
  exit 1
fi
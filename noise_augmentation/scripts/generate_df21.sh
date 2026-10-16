#!/usr/bin/env bash
set -euo pipefail

# Ctrl+C/터미널 종료 시 하위 프로세스까지 정리
trap 'echo "[TRAP] stopping all..."; kill $(jobs -pr) 2>/dev/null || true' INT TERM EXIT

# tqdm 실시간 출력
export PYTHONUNBUFFERED=1

# ---- 경로/환경 설정 ----
source ~/miniconda3/etc/profile.d/conda.sh

TOTAL_CORES=$(nproc)
GEN_CORES=$(( TOTAL_CORES*7/10 ))
AT_CORES=$(( TOTAL_CORES - GEN_CORES ))
GEN_CORES=$(( GEN_CORES>0 ? GEN_CORES : 1 ))
AT_CORES=$(( AT_CORES>0 ? AT_CORES : 1 ))
echo "[INFO] Cores: total=$TOTAL_CORES → general=$GEN_CORES, autotune=$AT_CORES"

# ====== DF21 전용 경로 ======
OUT_ROOT="${PROJECT_ROOT}/Dataset/icassp/df21"   # ✅ 결과물 루트
AUG_CFG="${PROJECT_ROOT}/Dataset/NoiseDataset/augmentation_config.yaml"
ASSIGN_DIR="${OUT_ROOT}/_assigned"
LOG_DIR="${OUT_ROOT}/_logs"
mkdir -p "$ASSIGN_DIR" "$LOG_DIR"

GEN_SCRIPT="${PROJECT_ROOT}/Dataset/NoiseDataset/generate_audio.py"
AT_SCRIPT="${PROJECT_ROOT}/Dataset/NoiseDataset/generate_audio_autotune.py"

# ✅ 네가 만든 DF21 프로토콜(형식: "eval /abs/path/to/DF_E_*.flac spoof")
PROTO_DF21="${PROJECT_ROOT}/protocols/df21.txt"   # ← 경로만 바꿔줘

SEED="20250909"
SR=16000
SUBSETS=(eval)  # DF21은 eval-only

ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "======================================"
echo "$(ts) [STEP 0] DF21 균등 샤드 생성 (bash-only)"
echo "→ ${ASSIGN_DIR}"
echo "======================================"

# STEP 0: 균등 샤드 생성 (clean/auto_tune 포함, 파이썬 코드 수정 필요 없음)
conda run -n dataset --no-capture-output python - <<PY
import os, random, re
from pathlib import Path

assign_dir = Path("${ASSIGN_DIR}")
assign_dir.mkdir(parents=True, exist_ok=True)
rng = random.Random(${SEED})

# DF21은 eval만
subset = "eval"

# 증강 목록(오토튠/클린 포함)
AUGS = [
    "background_noise","background_music","gaussian_noise",
    "band_pass_filter","pitch_shift",
    "time_stretch","echo","reverberation",
    "clean","auto_tune",
]

# 프로토콜 읽기
lines = []
with open("${PROTO_DF21}", "r", encoding="utf-8") as f:
    for s in f:
        s = s.strip()
        if not s or s.startswith("#"): continue
        parts = s.split()
        # 허용 포맷: "eval /abs/path spoof" 또는 "speaker /abs/path label"
        if len(parts) < 3: continue
        lines.append(s)

N = len(lines)
rng.shuffle(lines)

K = len(AUGS)
base, rem = divmod(N, K)
targets = [base + (1 if i < rem else 0) for i in range(K)]

outdir = assign_dir / subset
outdir.mkdir(parents=True, exist_ok=True)

idx = 0
total_dumped = 0
for i, aug in enumerate(AUGS):
    take = targets[i]
    shard = lines[idx:idx+take]
    idx += take
    outp = outdir / f"{aug}.txt"
    with open(outp, "w", encoding="utf-8") as f:
        for r in shard: f.write(r + "\n")
    print(f"[DUMP] {subset}/{aug}: {len(shard)} -> {outp}")
    total_dumped += len(shard)

with open(assign_dir / "_summary.txt", "w", encoding="utf-8") as f:
    for aug in AUGS:
        p = outdir / f"{aug}.txt"
        n = sum(1 for _ in open(p, "r", encoding="utf-8")) if p.exists() else 0
        f.write(f"{subset}/{aug}: {n}\n")
    f.write(f"TOTAL_DUMPED: {total_dumped}\n")
    f.write(f"TOTAL_LINES: {N}\n")

print(f"[DUMP][DONE] -> {assign_dir}")
PY
2>&1 | tee -a "$LOG_DIR/step0_assign_$(date +%Y%m%d_%H%M%S).log"

echo "$(ts) [STEP 0 DONE] 분배 파일 목록:"
find "$ASSIGN_DIR" -type f -name "*.txt" | sed 's/^/ - /'

# STEP 1: 일반 증강 (clean 샤드는 clean만, 나머지는 해당 증강만)
run_general_noise() {
  echo "$(ts) [STEP 1] DF21 일반 증강 시작"

  local AUGS=(background_noise background_music gaussian_noise band_pass_filter pitch_shift time_stretch echo reverberation)

  local SUB_DIR="${ASSIGN_DIR}/eval"
  [[ -d "$SUB_DIR" ]] || { echo "[INFO] skip eval (no shards)"; return 0; }

  mapfile -t SHARDS < <(find "$SUB_DIR" -maxdepth 1 -type f -name "*.txt" ! -name "auto_tune.txt" | sort)
  if [[ ${#SHARDS[@]} -eq 0 ]]; then
    echo "[INFO]   no general shards in eval"
    return 0
  fi

  for F in "${SHARDS[@]}"; do
    AUG_NAME="$(basename "${F%.txt}")"
    LINES=$(wc -l < "$F" || echo 0)
    echo "$(ts)    - eval/${AUG_NAME}: ${LINES} lines"

    if [[ "$AUG_NAME" == "clean" ]]; then
      # CLEAN 전용: 모든 증강 제외 + --copy-clean
      EXCLUDE_LIST=( "${AUGS[@]}" )
      conda run -n dataset --no-capture-output \
        python "$GEN_SCRIPT" \
          --protocol "$F" \
          --out-root "$OUT_ROOT" \
          --aug-config "$AUG_CFG" \
          --sr "$SR" \
          --copy-clean \
          --skip-existing \
          --force-subset eval \
          --num-workers "$GEN_CORES" \
          --meta-out meta_noise_df21.csv \
          --exclude "${EXCLUDE_LIST[@]}" \
      |& tee -a "$LOG_DIR/step1_eval_clean_$(date +%Y%m%d_%H%M%S).log"
    else
      # 해당 증강만: 다른 증강 exclude, clean은 기록/복사하지 않음
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
          --skip-existing \
          --force-subset eval \
          --num-workers "$GEN_CORES" \
          --meta-out meta_noise_df21.csv \
          --exclude "${EXCLUDE_LIST[@]}" \
      |& tee -a "$LOG_DIR/step1_eval_${AUG_NAME}_$(date +%Y%m%d_%H%M%S).log"
    fi
  done

  echo "$(ts) [STEP 1 DONE] DF21 일반 증강 완료"
}

# STEP 2: 오토튠
run_autotune() {
  echo "$(ts) [STEP 2] DF21 오토튠 시작"

  local AT_PROTO="${ASSIGN_DIR}/eval/auto_tune.txt"
  if [[ ! -f "$AT_PROTO" ]]; then
    echo "[INFO]  skip eval/auto_tune (no shard)"
    return 0
  fi
  LINES=$(wc -l < "$AT_PROTO" || echo 0)
  echo "$(ts)   - eval/auto_tune: ${LINES} lines"

  conda run -n auto-tune --no-capture-output \
    python "$AT_SCRIPT" \
      --protocol "$AT_PROTO" \
      --out-root "$OUT_ROOT" \
      --aug-config "$AUG_CFG" \
      --sr "$SR" \
      --copy-clean \
      --skip-existing \
      --force-subset eval \
      --meta-out meta_noise_autotune_df21.csv \
  |& tee -a "$LOG_DIR/step2_eval_autotune_$(date +%Y%m%d_%H%M%S).log"

  echo "$(ts) [STEP 2 DONE] DF21 오토튠 완료"
}

echo "======================================"
echo "$(ts) 병렬 실행 시작 (STEP 1 & STEP 2)"
echo "======================================"
run_general_noise & PID1=$!
run_autotune    & PID2=$!

echo "$(ts) 두 작업 종료 대기…"
wait $PID1; EXIT1=$?
wait $PID2; EXIT2=$?

echo "======================================"
if [[ $EXIT1 -eq 0 && $EXIT2 -eq 0 ]]; then
  echo "$(ts) [ALL DONE] DF21 일반+오토튠 성공"

  # 메타 병합 (변수 치환 되도록 <<PY 사용)
  conda run -n dataset --no-capture-output python - <<PY
import os, pandas as pd
root = os.path.expandvars("${OUT_ROOT}")
A = os.path.join(root, "meta_noise_df21.csv")
B = os.path.join(root, "meta_noise_autotune_df21.csv")
A_df = pd.read_csv(A) if os.path.exists(A) else pd.DataFrame()
B_df = pd.read_csv(B) if os.path.exists(B) else pd.DataFrame()
df = pd.concat([A_df, B_df], ignore_index=True)
if {"label1","file_path"} <= set(df.columns):
    df = df.drop_duplicates(subset=["label1","file_path"])
outp = os.path.join(root, "meta_noise_all_df21.csv")
df.to_csv(outp, index=False)
print(f"[INFO] merged meta -> {outp} ({len(df)} rows)")
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
  tail -n 50 "$LOG_DIR"/step1_* 2>/dev/null || true
  echo "---- Last 50 lines (autotune) ----"
  tail -n 50 "$LOG_DIR"/step2_* 2>/dev/null || true
  echo "======================================"
  exit 1
fi

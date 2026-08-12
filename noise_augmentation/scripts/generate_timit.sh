#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh

# Detect number of CPU cores for optimal parallelization
NUM_WORKERS=$(nproc)
echo "Detected $NUM_WORKERS CPU cores - using parallel processing"

###############################################
# Run both steps in parallel for maximum efficiency
###############################################
echo "======================================"
echo "Starting parallel dataset generation:"
echo "[STEP 1] General noise dataset (background)"
echo "[STEP 2] Auto-tune dataset (background)"
echo "======================================"

# Function to run general noise dataset generation
run_general_noise() {
    echo "[STEP 1] Activating conda env: dataset"
    conda activate dataset
    
    python ${PROJECT_ROOT}/Dataset/NoiseDataset/generate_audio.py \
      --protocol \
        ${PROJECT_ROOT}/Dataset/TIMIT/timit_eval.txt \
      --out-root ${PROJECT_ROOT}/Dataset/NoiseDataset/TIMIT \
      --aug-config ${PROJECT_ROOT}/Dataset/NoiseDataset/augmentation_config.yaml \
      --sr 16000 \
      --copy-clean \
      --num-workers $NUM_WORKERS \
      --force-subset eval
    
    conda deactivate
    echo "[STEP 1 DONE] General noise dataset generated."
}

# Function to run auto-tune dataset generation
run_autotune() {
    echo "[STEP 2] Activating conda env: auto-tune"
    conda activate auto-tune
    
    python ${PROJECT_ROOT}/Dataset/NoiseDataset/generate_audio_autotune.py \
      --protocol \
        ${PROJECT_ROOT}/Dataset/TIMIT/timit_eval.txt \
      --out-root ${PROJECT_ROOT}/Dataset/NoiseDataset/TIMIT \
      --aug-config ${PROJECT_ROOT}/Dataset/NoiseDataset/augmentation_config.yaml \
      --sr 16000 \
      --copy-clean \
      --meta-out meta_noise_autotune.csv \
      --force-subset eval
    
    conda deactivate
    echo "[STEP 2 DONE] Auto-tune dataset generated."
}

# Export functions to make them available to background processes
export -f run_general_noise
export -f run_autotune

# Run both steps in parallel
echo "Starting both processes in parallel..."
run_general_noise &
PID1=$!

run_autotune &
PID2=$!

# Wait for both to complete
echo "Waiting for both processes to complete..."
wait $PID1
EXIT1=$?
wait $PID2  
EXIT2=$?

# Check results
if [ $EXIT1 -eq 0 ] && [ $EXIT2 -eq 0 ]; then
    echo "======================================"
    echo "[ALL DONE] Noise dataset (general + auto_tune) successfully generated in parallel!"
    echo "======================================"
else
    echo "======================================"
    echo "[ERROR] One or both processes failed:"
    echo "General noise exit code: $EXIT1"
    echo "Auto-tune exit code: $EXIT2"
    echo "======================================"
    exit 1
fi

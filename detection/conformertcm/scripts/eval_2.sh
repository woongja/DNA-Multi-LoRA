#!/bin/bash

#######################
# 1라운드 (ASV19 baseline + finetune)
#######################
# CUDA_VISIBLE_DEVICES=2 python main.py \
#   --protocols_path ${PROJECT_ROOT}/protocols/protocol_asv19.txt \
#   --database_path ${PROJECT_ROOT}/Dataset/icassp/asv19 \
#   --average_model false \
#   --model_path ${PROJECT_ROOT}/tcm_add/avg_5_best.pth \
#   --eval_output ${PROJECT_ROOT}/results/final/conformertcm-baseline-asv19.txt &

CUDA_VISIBLE_DEVICES=0 python main.py \
  --protocols_path ${PROJECT_ROOT}/protocols/protocol_asv19.txt \
  --database_path ${PROJECT_ROOT}/Dataset/icassp/asv19 \
  --average_model false \
  --model_path ${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g7.pth \
  --eval_output ${PROJECT_ROOT}/results/FT-seperate/conformertcm-finetune-asv19.txt &

# echo "=== 1라운드 완료 ==="


#######################
# 2라운드 (DF21 baseline + finetune)
#######################
# CUDA_VISIBLE_DEVICES=2 python main.py \
#   --protocols_path ${PROJECT_ROOT}/protocols/protocol_df21.txt \
#   --database_path ${PROJECT_ROOT}/Dataset/icassp/df21 \
#   --average_model false \
#   --model_path ${PROJECT_ROOT}/tcm_add/avg_5_best.pth \
#   --eval_output ${PROJECT_ROOT}/results/final/conformertcm-baseline-df21.txt &

CUDA_VISIBLE_DEVICES=1 python main.py \
  --protocols_path ${PROJECT_ROOT}/protocols/protocol_df21.txt \
  --database_path ${PROJECT_ROOT}/Dataset/icassp/df21 \
  --average_model false \
  --model_path ${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g7.pth \
  --eval_output ${PROJECT_ROOT}/results/FT-seperate/conformertcm-finetune-df21.txt &

# echo "=== 2라운드 완료 ==="


#######################
# 3라운드 (ADD2022 baseline + finetune)
#######################
# CUDA_VISIBLE_DEVICES=2 python main.py \
#   --protocols_path ${PROJECT_ROOT}/protocols/protocol_add2022.txt \
#   --database_path ${PROJECT_ROOT}/Dataset/hung \
#   --average_model false \
#   --model_path ${PROJECT_ROOT}/tcm_add/avg_5_best.pth \
#   --eval_output ${PROJECT_ROOT}/results/final/conformertcm-baseline-add2022.txt &

CUDA_VISIBLE_DEVICES=2 python main.py \
  --protocols_path ${PROJECT_ROOT}/protocols/protocol_add2022.txt \
  --database_path ${PROJECT_ROOT}/Dataset/hung \
  --average_model false \
  --model_path ${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g7.pth \
  --eval_output ${PROJECT_ROOT}/results/FT-seperate/conformertcm-finetune-add2022.txt &

# echo "=== 3라운드 완료 ==="


#######################
# 4라운드 (ITW baseline + finetune)
#######################
# CUDA_VISIBLE_DEVICES=2 python main.py \
#   --protocols_path ${PROJECT_ROOT}/protocols/protocol_itw.txt \
#   --database_path /data/Datasets/itw \
#   --average_model false \
#   --model_path ${PROJECT_ROOT}/tcm_add/avg_5_best.pth \
#   --eval_output ${PROJECT_ROOT}/results/final/conformertcm-baseline-itw.txt &

CUDA_VISIBLE_DEVICES=2 python main.py \
  --protocols_path ${PROJECT_ROOT}/protocols/protocol_itw.txt \
  --database_path /data/Datasets/itw \
  --average_model false \
  --model_path ${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g7.pth \
  --eval_output ${PROJECT_ROOT}/results/FT-seperate/conformertcm-finetune-itw.txt &

echo "=== 4라운드 완료 ==="

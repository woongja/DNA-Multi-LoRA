#!/bin/bash

#######################
# 1라운드 (ASV19 baseline + finetune)
#######################
# CUDA_VISIBLE_DEVICES=0 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/Best_LA_model_for_DF.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/protocol_asv19.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/final/aasist-baseline-asv19.txt' &

CUDA_VISIBLE_DEVICES=1 python main_SSL_DF.py \
  --track=DF \
  --is_eval \
  --eval \
  --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/best_model.pth' \
  --protocols_path '${PROJECT_ROOT}/protocols/noise_type_asv19/clean.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --eval_output '${PROJECT_ROOT}/results/FT-aug/aasist-finetune-asv19-clean.txt' 

echo "=== 1라운드 완료 ==="

#######################
# 3라운드 (ADD2022 baseline + finetune)
#######################
# CUDA_VISIBLE_DEVICES=0 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/Best_LA_model_for_DF.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/protocol_add2022.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/hung' \
#   --eval_output '${PROJECT_ROOT}/results/final/aasist-baseline-add2022.txt' &

# CUDA_VISIBLE_DEVICES=1 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/best_model.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/protocol_add2022.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/hung' \
#   --eval_output '${PROJECT_ROOT}/results/FT-aug/aasist-finetune-add2022.txt' &

# echo "=== 3라운드 완료 ==="


#######################
# 4라운드 (ITW baseline + finetune)
#######################
# CUDA_VISIBLE_DEVICES=0 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/Best_LA_model_for_DF.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/protocol_itw.txt' \
#   --database_path '/data/Datasets/itw' \
#   --eval_output '${PROJECT_ROOT}/results/final/aasist-baseline-itw.txt' &

# CUDA_VISIBLE_DEVICES=2 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/best_model.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/protocol_itw.txt' \
#   --database_path '/data/Datasets/itw' \
#   --eval_output '${PROJECT_ROOT}/results/FT-aug/aasist-finetune-itw.txt' &

#######################
# 2라운드 (DF21 baseline + finetune)
#######################
# CUDA_VISIBLE_DEVICES=0 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/Best_LA_model_for_DF.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/protocol_df21.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/df21' \
#   --eval_output '${PROJECT_ROOT}/results/final/aasist-baseline-df21.txt' &

# CUDA_VISIBLE_DEVICES=2 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/best_model.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/protocol_df21.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/df21' \
#   --eval_output '${PROJECT_ROOT}/results/FT-aug/aasist-finetune-df21.txt'

# echo "=== 2라운드 완료 ==="
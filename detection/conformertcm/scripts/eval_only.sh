CUDA_VISIBLE_DEVICES=0 python main.py \
  --protocols_path ${PROJECT_ROOT}/protocols/noise_type_asv19/manipulation.txt \
  --database_path ${PROJECT_ROOT}/Dataset/icassp/asv19 \
  --average_model false \
  --model_path ${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g7.pth \
  --eval_output ${PROJECT_ROOT}/results/FT-aug/conformertcm-finetune-mani.txt
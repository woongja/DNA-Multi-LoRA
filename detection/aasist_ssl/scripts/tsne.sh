CUDA_VISIBLE_DEVICES=0 python main_SSL_DF_embedding.py \
  --database_path ${PROJECT_ROOT}/Dataset/icassp/asv19 \
  --protocols_path ${PROJECT_ROOT}/protocols/noise_type_asv19/manipulation.txt \
  --model_path ${PROJECT_ROOT}/SSL_Anti-spoofing/Best_LA_model_for_DF.pth \
  --dump_dev_embeddings \
  --embedding_dir ${PREPROCESS_DIR}/embeddings/g5/aasist/baseline &

CUDA_VISIBLE_DEVICES=1 python main_SSL_DF_embedding.py \
  --database_path ${PROJECT_ROOT}/Dataset/icassp/asv19 \
  --protocols_path ${PROJECT_ROOT}/protocols/noise_type_asv19/manipulation.txt \
  --model_path ${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g7.pth \
  --dump_dev_embeddings \
  --embedding_dir ${PREPROCESS_DIR}/embeddings/g5/aasist/ft

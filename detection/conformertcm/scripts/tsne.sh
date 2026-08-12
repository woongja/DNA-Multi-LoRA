CUDA_VISIBLE_DEVICES=2 python main_embedding.py \
    --train='true' \
    --batch_size=20 \
    --model_path='${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g7.pth' \
    --database_path='${PROJECT_ROOT}/Dataset/icassp/asv19' \
    --protocols_path='${PROJECT_ROOT}/protocols/noise_type_asv19/manipulation.txt' \
    --num_epochs=10 \
    --dump_dev_embeddings \
    --embedding_dir '${PREPROCESS_DIR}/embeddings/g5/conformertcm/ft' &

CUDA_VISIBLE_DEVICES=2 python main_embedding.py \
    --train='true' \
    --batch_size=20 \
    --model_path='${PROJECT_ROOT}/tcm_add/avg_5_best.pth' \
    --database_path='${PROJECT_ROOT}/Dataset/icassp/asv19' \
    --protocols_path='${PROJECT_ROOT}/protocols/noise_type_asv19/manipulation.txt' \
    --num_epochs=10 \
    --dump_dev_embeddings \
    --embedding_dir '${PREPROCESS_DIR}/embeddings/g5/conformertcm/baseline'
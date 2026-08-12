CUDA_VISIBLE_DEVICES=2 python main.py \
    --train='true' \
    --batch_size=20 \
    --model_path='${PROJECT_ROOT}/tcm_add/avg_5_best.pth' \
    --database_path='${PROJECT_ROOT}/Dataset/icassp/asv19' \
    --protocols_path='${PROJECT_ROOT}/protocols/protocol_asv19_aug.txt' \
    --num_epochs=30 \
    --early_stop_patience=3
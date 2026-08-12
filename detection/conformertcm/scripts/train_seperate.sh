CUDA_VISIBLE_DEVICES=0 python main.py \
    --train='true' \
    --batch_size=20 \
    --model_path='${PROJECT_ROOT}/tcm_add/avg_5_best.pth' \
    --database_path='${PROJECT_ROOT}/Dataset/icassp/asv19' \
    --protocols_path='${PROJECT_ROOT}/protocols/noise_type_asv19/background_music_noise.txt' \
    --num_epochs=30 \
    --early_stop_patience=3 \
    --model_name=model_g1.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --train='true' \
    --batch_size=20 \
    --model_path='${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g1.pth' \
    --database_path='${PROJECT_ROOT}/Dataset/icassp/asv19' \
    --protocols_path='${PROJECT_ROOT}/protocols/noise_type_asv19/auto_tune.txt' \
    --num_epochs=30 \
    --early_stop_patience=3 \
    --model_name=model_g2.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --train='true' \
    --batch_size=20 \
    --model_path='${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g2.pth' \
    --database_path='${PROJECT_ROOT}/Dataset/icassp/asv19' \
    --protocols_path='${PROJECT_ROOT}/protocols/noise_type_asv19/band_pass_filter.txt' \
    --num_epochs=30 \
    --early_stop_patience=3 \
    --model_name=model_g3.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --train='true' \
    --batch_size=20 \
    --model_path='${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g3.pth' \
    --database_path='${PROJECT_ROOT}/Dataset/icassp/asv19' \
    --protocols_path='${PROJECT_ROOT}/protocols/noise_type_asv19/echo.txt' \
    --num_epochs=30 \
    --early_stop_patience=3 \
    --model_name=model_g4.pth


CUDA_VISIBLE_DEVICES=0 python main.py \
    --train='true' \
    --batch_size=20 \
    --model_path='${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g4.pth' \
    --database_path='${PROJECT_ROOT}/Dataset/icassp/asv19' \
    --protocols_path='${PROJECT_ROOT}/protocols/noise_type_asv19/manipulation.txt' \
    --num_epochs=30 \
    --early_stop_patience=3 \
    --model_name=model_g5.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --train='true' \
    --batch_size=20 \
    --model_path='${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g5.pth' \
    --database_path='${PROJECT_ROOT}/Dataset/icassp/asv19' \
    --protocols_path='${PROJECT_ROOT}/protocols/noise_type_asv19/gaussian_noise.txt' \
    --num_epochs=30 \
    --early_stop_patience=3 \
    --model_name=model_g6.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --train='true' \
    --batch_size=20 \
    --model_path='${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g6.pth' \
    --database_path='${PROJECT_ROOT}/Dataset/icassp/asv19' \
    --protocols_path='${PROJECT_ROOT}/protocols/noise_type_asv19/reverberation.txt' \
    --num_epochs=30 \
    --early_stop_patience=3 \
    --model_name=model_g7.pth
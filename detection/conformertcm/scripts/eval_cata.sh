CUDA_VISIBLE_DEVICES=0 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g0.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/avg_5_best.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-baseline-g0.txt' &

CUDA_VISIBLE_DEVICES=0 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g0.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g1.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g1-g0.txt' &

CUDA_VISIBLE_DEVICES=0 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g0.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g2.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g2-g0.txt' &

CUDA_VISIBLE_DEVICES=1 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g1.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g2.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g2-g1.txt' &

CUDA_VISIBLE_DEVICES=1 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g0.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g3.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g3-g0.txt' &

CUDA_VISIBLE_DEVICES=1 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g1.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g3.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g3-g1.txt' &

CUDA_VISIBLE_DEVICES=2 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g2.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g3.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g3-g2.txt' &

CUDA_VISIBLE_DEVICES=2 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g0.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g4.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g4-g0.txt' &

CUDA_VISIBLE_DEVICES=2 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g1.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g4.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g4-g1.txt'

CUDA_VISIBLE_DEVICES=0 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g2.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g4.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g4-g2.txt' &

CUDA_VISIBLE_DEVICES=0 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g3.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g4.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g4-g3.txt' &

CUDA_VISIBLE_DEVICES=0 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g0.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g5.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g5-g0.txt' &

CUDA_VISIBLE_DEVICES=1 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g1.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g5.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g5-g1.txt' &

CUDA_VISIBLE_DEVICES=1 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g2.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g5.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g5-g2.txt' &

CUDA_VISIBLE_DEVICES=1 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g3.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g5.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g5-g3.txt' &

CUDA_VISIBLE_DEVICES=2 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g4.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g5.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g5-g4.txt' &

CUDA_VISIBLE_DEVICES=2 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g0.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g6.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g6-g0.txt' &

CUDA_VISIBLE_DEVICES=2 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g1.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g6.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g6-g1.txt'

CUDA_VISIBLE_DEVICES=0 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g2.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g6.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g6-g2.txt' &

CUDA_VISIBLE_DEVICES=0 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g3.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g6.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g6-g3.txt' &

CUDA_VISIBLE_DEVICES=0 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g4.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g6.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g6-g4.txt' &

CUDA_VISIBLE_DEVICES=1 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g5.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g6.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g6-g5.txt'

CUDA_VISIBLE_DEVICES=1 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g0.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g7.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g7-g0.txt' &

CUDA_VISIBLE_DEVICES=1 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g1.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g7.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g7-g1.txt' &

CUDA_VISIBLE_DEVICES=2 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g2.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g7.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g7-g2.txt' &

CUDA_VISIBLE_DEVICES=2 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g3.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g7.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g7-g3.txt' &

CUDA_VISIBLE_DEVICES=2 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g4.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g7.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g7-g4.txt'

CUDA_VISIBLE_DEVICES=0 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g5.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g7.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g7-g5.txt' &

CUDA_VISIBLE_DEVICES=1 python main.py \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g6.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --average_model false \
  --model_path '${PROJECT_ROOT}/tcm_add/models/FT-seperate/model_g7.pth' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/conformertcm-g7-g6.txt'


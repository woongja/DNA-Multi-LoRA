# CUDA_VISIBLE_DEVICES=0 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/best_model.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g0.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-baseline-g0.txt' &

# CUDA_VISIBLE_DEVICES=0 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g1.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g0.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g1-g0.txt' &

# CUDA_VISIBLE_DEVICES=1 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g2.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g0.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g2-g0.txt' &

# CUDA_VISIBLE_DEVICES=1 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g2.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g1.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g2-g1.txt' &

# CUDA_VISIBLE_DEVICES=2 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g3.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g0.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g3-g0.txt' &

# CUDA_VISIBLE_DEVICES=2 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g3.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g1.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g3-g1.txt'

# CUDA_VISIBLE_DEVICES=0 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g3.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g2.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g3-g2.txt' &

# CUDA_VISIBLE_DEVICES=0 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g4.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g0.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g4-g0.txt' &

# CUDA_VISIBLE_DEVICES=1 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g4.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g1.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g4-g1.txt' &

# CUDA_VISIBLE_DEVICES=1 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g4.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g2.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g4-g2.txt' &

# CUDA_VISIBLE_DEVICES=2 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g4.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g3.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g4-g3.txt' &

# CUDA_VISIBLE_DEVICES=2 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g5.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g0.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g5-g0.txt'

# CUDA_VISIBLE_DEVICES=0 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g5.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g1.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g5-g1.txt' &

# CUDA_VISIBLE_DEVICES=0 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g5.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g2.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g5-g2.txt' &

# CUDA_VISIBLE_DEVICES=0 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g5.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g3.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g5-g3.txt' &

# CUDA_VISIBLE_DEVICES=1 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g5.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g4.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g5-g4.txt' &

# CUDA_VISIBLE_DEVICES=1 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g6.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g0.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g6-g0.txt' &

# CUDA_VISIBLE_DEVICES=2 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g6.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g1.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g6-g1.txt'

# CUDA_VISIBLE_DEVICES=1 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g6.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g2.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g6-g2.txt'

# CUDA_VISIBLE_DEVICES=0 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g6.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g3.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g6-g3.txt' &

# CUDA_VISIBLE_DEVICES=1 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g6.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g4.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g6-g4.txt' &

# CUDA_VISIBLE_DEVICES=1 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g6.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g5.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g6-g5.txt' &

# CUDA_VISIBLE_DEVICES=2 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g7.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g0.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g7-g0.txt' &

# CUDA_VISIBLE_DEVICES=2 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g7.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g1.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g7-g1.txt'

CUDA_VISIBLE_DEVICES=0 python main_SSL_DF.py \
  --track=DF \
  --is_eval \
  --eval \
  --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g7.pth' \
  --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g2.txt' \
  --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
  --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g7-g2.txt'

# CUDA_VISIBLE_DEVICES=0 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g7.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g3.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g7-g3.txt' &

# CUDA_VISIBLE_DEVICES=2 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g7.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g4.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g7-g4.txt' &

# CUDA_VISIBLE_DEVICES=2 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g7.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g5.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g7-g5.txt'

# CUDA_VISIBLE_DEVICES=2 python main_SSL_DF.py \
#   --track=DF \
#   --is_eval \
#   --eval \
#   --model_path '${PROJECT_ROOT}/SSL_Anti-spoofing/models/model_DF_WCE_30_14_1e-06/model_g7.pth' \
#   --protocols_path '${PROJECT_ROOT}/protocols/catastropy/g6.txt' \
#   --database_path '${PROJECT_ROOT}/Dataset/icassp/asv19' \
#   --eval_output '${PROJECT_ROOT}/results/FT-cata/aasist-g7-g6.txt' &
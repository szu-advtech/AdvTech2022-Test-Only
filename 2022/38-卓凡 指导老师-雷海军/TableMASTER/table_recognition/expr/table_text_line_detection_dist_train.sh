# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 ./tools/dist_train.sh ./configs/textdet/psenet/psenet_r50_fpnf_600e_pubtabnet.py ./work_dir/1210_PseNet_textdet 8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 PORT=29500 ./tools/dist_train.sh ./configs/textdet/psenet/psenet_r50_fpnf_600e_pubtabnet.py ./work_dir/1210_PseNet_textdet 7

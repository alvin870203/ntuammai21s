mkdir 'log'
python train.py \
    --data_root '/home/chihyuan/ntuammai21s/hw1/project/data/train/C_prep' \
    --train_file '/home/chihyuan/ntuammai21s/hw1/project/data/files/train_list.txt' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file '../backbone_conf.yaml' \
    --head_type 'AM-Softmax' \
    --head_conf_file '../head_conf.yaml' \
    --lr 0.1 \
    --out_dir 'out_dir' \
    --epoches 200 \
    --step '80, 140, 180' \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 128 \
    --momentum 0.9 \
    --log_dir 'log' \
    --tensorboardx_logdir 'mv-hrnet' \
    2>&1 | tee log/log.log

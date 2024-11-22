#!/usr/bin/env bash
python test.py \
    --backbone resnet101 \
    --epochs 7 \
    --eval  \
    --num_feature_levels 1 \
    --num_queries 300 \
    --dilation \
    --batch_size 1 \
    --num_ref_frames 14 \
    --resume exps/multibaseline/r101_e12_class2/checkpoint0003.pth \
    --lr_drop_epochs 4 6 \
    --num_workers 16 \
    --with_box_refine \
    --dataset_file vid_multi_mine_multi \
    --output_dir exps/multibaseline/r101_e12_class2
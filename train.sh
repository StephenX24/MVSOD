#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`
EXP_DIR=exps/multibaseline/r101_e12_class2
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone resnet101 \
    --epochs 12 \
    --num_feature_levels 1 \
    --num_queries 300 \
    --dilation \
    --batch_size 1 \
    --num_ref_frames 14 \
    --resume checkpoint0020.pth \
    --lr 1e-5 \
    --lr_backbone 1e-6 \
    --num_workers 8 \
    --with_box_refine \
    --coco_pretrain \
    --dataset_file vid_multi_mine_multi \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
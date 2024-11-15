#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/singlebaseline/r50_e10_class2_1114
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone resnet50 \
    --epochs 10 \
    --num_feature_levels 1 \
    --num_queries 300 \
    --dilation \
    --batch_size 1 \
    --num_ref_frames 14 \
    --resume r50_deformable_detr_single_scale_dc5-checkpoint.pth \
    --lr_drop_epochs 7 9 \
    --num_workers 8 \
    --with_box_refine \
    --coco_pretrain \
    --dataset_file vid_multi_mine_multi \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T


# EXP_DIR=exps/singlebaseline/r50_e15_class91_1114
# mkdir -p ${EXP_DIR}
# PY_ARGS=${@:1}
# python -u main.py \
#     --backbone resnet50 \
#     --epochs 15 \
#     --num_feature_levels 1 \
#     --num_queries 300 \
#     --dilation \
#     --batch_size 1 \
#     --num_ref_frames 14 \
#     --resume r50_deformable_detr_single_scale_dc5-checkpoint.pth \
#     --lr_drop_epochs 9 12 \
#     --num_workers 8 \
#     --with_box_refine \
#     --coco_pretrain \
#     --dataset_file vid_multi_mine_multi \
#     --output_dir ${EXP_DIR} \
#     ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
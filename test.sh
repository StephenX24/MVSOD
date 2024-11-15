# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# python tools/launch.py --nnodes 1 --node_rank 0 --master_addr 127.0.0.2 --master_port 2000 --nproc_per_node 8 configs/r101_eval_multi_mine_multi.sh

# python test.py  --backbone resnet50 \
#                 --epochs 6 \
#                 --eval  \
#                 --num_feature_levels 1 \
#                 --num_queries 300 \
#                 --dilation \
#                 --batch_size 1 \
#                 --num_ref_frames 14 \
#                 --resume exps/multibaseline/r50_grad/e7_class2/checkpoint0006.pth \
#                 --lr_drop_epochs 4 5 \
#                 --num_workers 16 \
#                 --with_box_refine \
#                 --dataset_file vid_multi_mine_multi \
#                 --output_dir exps/multibaseline/r50_grad/e7_class2/exps_multi

python test.py  --backbone resnet101 \
                --epochs 7 \
                --eval  \
                --num_feature_levels 1 \
                --num_queries 300 \
                --dilation \
                --batch_size 1 \
                --num_ref_frames 14 \
                --resume checkpoint0020.pth \
                --lr_drop_epochs 4 6 \
                --num_workers 16 \
                --with_box_refine \
                --dataset_file vid_multi_mine_multi \
                --output_dir exps/multibaseline/r50_grad/e7_class2/exps_multi
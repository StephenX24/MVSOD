# python main.py --backbone resnet50 \
#                --epochs 10 \
#                --eval \
#                --num_feature_levels 1 \
#                --num_queries 300 \
#                --dilation \
#                --batch_size 1 \
#                --num_ref_frames 14 \
#                --resume exps/singlebaseline/r50_e10_class2_1114/checkpoint0009.pth \
#                --lr_drop_epochs 7 9 \
#                --num_workers 16 \
#                --with_box_refine \
#                --dataset_file vid_multi_mine_multi \
#                --output_dir exps/multibaseline/r50_grad/e7_class2/exps_multi



python main.py --backbone resnet101 \
               --epochs 7 \
               --eval \
               --num_feature_levels 1 \
               --num_queries 300 \
               --dilation \
               --batch_size 1 \
               --num_ref_frames 14 \
               --resume exps/singlebaseline/r101_e10_class2_1115/checkpoint0004.pth \
               --lr_drop_epochs 4 6 \
               --num_workers 16 \
               --with_box_refine \
               --dataset_file vid_multi_mine_multi \
               --output_dir exps/multibaseline/r50_grad/e7_class2/exps_multi
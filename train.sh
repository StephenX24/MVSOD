export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tools/launch.py --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 3000 --nproc_per_node 8 configs/r101_train_multi_mine_multi.sh


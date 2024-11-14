## Install
  
* Python>=3.8

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n MVSOD python=3.8 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate MVSOD
    ```

* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh # or python setup.py build install
    ```

## Usage

### Dataset preparation

1. Please download sky_data3 from [here](https://pan.baidu.com/s/1qI1EjqF5ll7WbzC9H1VjLQ?pwd=1234). The numbers of RGB/IR images in the train, validation and test sets are 8782, 1999, and 1858. The [json](./tools/covert2coco.py) file can be genetated by './tools/covert2coco.py' And the path structure should be as follows:

```
project_root/
└── data/
    └── vid/
        ├── Data
            ├── sky_data3/
        └── annotations/
        	├── sky_data_vid_test.json
            ├── sky_data_vid_train.json
        	└── sky_data_vid_val.json

```

#### Training on multi-gpus

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tools/launch.py --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 3000 --nproc_per_node 8 configs/r101_train_multi_mine_multi.sh
```
1. Pretrained model from [here](https://pan.baidu.com/s/1G10qdtbkbDOZaAoIqiGWXQ?pwd=1234).
#### Training on single-gpus

```bash
python main.py --backbone resnet101 \
               --epochs 10 \
               --num_feature_levels 1 \
               --num_queries 300 \
                --dilation \
                --batch_size 1 \
                --num_ref_frames 14 \
                --resume exps/our_models/COCO_pretrained_model/r101_deformable_detr_single_scale_bbox_refinement-dc5_checkpoint0049.pth \
                --lr_drop_epochs 7 9 \
                --num_workers 8 \
                --with_box_refine \
                --coco_pretrain \
                --dataset_file vid_multi_mine_multi \
                --output_dir exps/singlebaseline/r101_e8_nf4_ld6,7_lr0.0002_nq300_bs4_wbox_joint_MEGA_detrNorm_class31_pretrain_coco_dc5
```

#### Validating on single-gpus

```bash
python main.py --backbone resnet101 \
               --epochs 7 \
               --eval \
               --num_feature_levels 1 \
               --num_queries 300 \
               --dilation \
               --batch_size 1 \
               --num_ref_frames 14 \
               --resume exps/multibaseline/r101_grad/e7_nf1_ld4,6_lr0.0002_nq300_wbox_MEGA_detrNorm_preSingle_nr14_dc5_nql3_filter150_75_40/checkpoint0020.pth \
               --lr_drop_epochs 4 6 \
               --num_workers 16 \
               --with_box_refine \
               --dataset_file vid_multi_mine_multi \
               --output_dir exps/our_models/exps_multi/r101_81.7
```

### Testing on single-gpus

```bash
python test.py  --backbone resnet101 \
                --epochs 7 \
                --eval  \
                --num_feature_levels 1 \
                --num_queries 300 \
                --dilation \
                --batch_size 1 \
                --num_ref_frames 14 \
                --resume exps/multibaseline/r101_grad/e7_nf1_ld4,6_lr0.0002_nq300_wbox_MEGA_detrNorm_preSingle_nr14_dc5_nql3_filter150_75_40/checkpoint0020.pth \
                --lr_drop_epochs 4 6 \
                --num_workers 16 \
                --with_box_refine \
                --dataset_file \
                vid_multi_mine_multi \
                --output_dir exps/our_models/exps_multi/r101_81.7
```
The you will get a 'test_save.json' file for evaluation.

## Acknowledgment
This code is based on TransVOD(https://github.com/SJTU-LuHe/TransVOD)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 @Time :  11:01 AM
 @Author : Zhao kunlong
 @Description: 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import json
import os
import cv2 as cv
from glob import glob
from collections import defaultdict
from tqdm import tqdm

CLASSES = ('uav',)

CLASSES_ENCODES = ('None',)

cats_id_maps = {}
for k, v in enumerate(CLASSES_ENCODES, 1):
    cats_id_maps[v] = k


def decode_gt(gts: str):
    ir_gt = gts.split('\n')[0].split(',')
    obj_num = int(ir_gt[1])
    frame_id = int(ir_gt[0])
    ir_gt_dict = {}
    for j in range(obj_num):
        obj_id = int(ir_gt[2 + j * 6])
        ir_gt_dict[obj_id] = [float(v) for v in ir_gt[3 + j * 6:7 + j * 6]]

    return ir_gt_dict, frame_id

def conver2coco(root, VID:defaultdict(list), split:str='train'):
    records = dict(
        vid_id=1,
        img_id=1,
        ann_id=1,
        global_instance_id=1,
        num_vid_train_frames=0,
        num_no_objects=0)

    trained_frames = 0
    videos = glob(os.path.join(root, split,'*'))
    pdr = tqdm(videos, desc='processing...')
    for _video in pdr:
        video_name = os.path.join(*_video.split('/')[-3:])

        vid_train_frames = []

        gt_path = os.path.join(_video, 'rgb_gt.txt')
        with open(gt_path, 'r') as f:
            rgb_gts = f.readlines()
        rgb_gt_dicts = {}
        for rgb_gt in rgb_gts:
            rgb_gt_dict, frame_id = decode_gt(rgb_gt)
            rgb_gt_dicts[frame_id] = rgb_gt_dict
            if len(rgb_gt_dict) != 0:
                vid_train_frames.append(frame_id)

        trained_frames += len(vid_train_frames)

        if split in ['val', 'val_train', 'test']:
            video = dict(
                id=records['vid_id'],
                name=video_name,
                vid_train_frames=[])
        else:
            video = dict(
                id=records['vid_id'],
                name=video_name,
                vid_train_frames=vid_train_frames)
        VID['videos'].append(video)

        for frame_id in vid_train_frames:
            image_path = os.path.join(_video, 'rgb', '{:06d}.jpg'.format(frame_id))
            try:
                img = cv.imread(image_path)
                size = img.shape
                height, width, _ = size
            except Exception as e:
                print(e)
                print(image_path)

            image = dict(
                file_name=os.path.join(*image_path.split('/')[-5:]),
                height=height,
                width=width,
                id=records['img_id'],
                frame_id=frame_id,
                video_id=records['vid_id'],
                is_vid_train_frame=True)
            VID['images'].append(image)

            if split != 'test ': 
                obj_annos = rgb_gt_dicts[frame_id]

                for obj_anno in obj_annos.values():
                    x1, y1, x2, y2 = obj_anno
                    w, h = x2 - x1, y2 - y1
                    x1, y1, w, h = int(x1), int(y1), int(w), int(h)

                    ann = dict(
                        id=records['ann_id'],
                        video_id=records['vid_id'],
                        image_id=records['img_id'],
                        category_id=1,
                        instance_id=-1,
                        bbox=[x1, y1, w, h],
                        area=w * h,
                        iscrowd=False,
                        occluded=False,
                        generated=False)
                    VID['annotations'].append(ann)
                    records['ann_id'] += 1
            records['img_id'] += 1
        records['vid_id'] += 1
    assert trained_frames == records['img_id'] - 1
    with open(f'sky_data_vid_{split}.json', 'w') as f:
        json.dump(VID, f)

if __name__ == '__main__':
    categories = []
    for k, v in enumerate(CLASSES, 1):
        categories.append(
            dict(id=k, name=v, encode_name=CLASSES_ENCODES[k - 1]))

    VID_train = defaultdict(list)
    VID_train['categories'] = categories
    # root = '/home/baode/project/data/sky_data1'
    root = '/home/baode/project/data/sky_data3/'

    # conver2coco(root, VID_train, split='train')
    conver2coco(root, VID_train, split='test')

# Modified by Lu He
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask
from .coco_video_parser import CocoVID
from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms_multi as T
from torch.utils.data.dataset import ConcatDataset
import random
import os
from PIL import Image
import os
import os.path
from io import BytesIO


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, interval1, interval2, num_ref_frames= 3,
        is_train = True,  filter_key_img=True,  cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.ann_file = ann_file
        self.frame_range = [-2, 2]
        self.num_ref_frames = num_ref_frames
        self.cocovid = CocoVID(self.ann_file)
        self.is_train = is_train
        self.filter_key_img = filter_key_img
        self.interval1 = interval1
        self.interval2 = interval2

    def get_image(self, path):
        if self.cache_mode:
            raise NotImplementedError
        rgb = Image.open(os.path.join(self.root, path)).convert('RGB')
        ir_path = path.split('/')
        ir_path[-2] = 'ir'
        ir_path = os.path.join(*ir_path)
        ir = Image.open(os.path.join(self.root, ir_path)).convert('RGB')
        return rgb, ir
    
    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        # idx若为675834，则img_id为675835(img_id=idx+1)
        imgs = []
        coco = self.coco
        img_id = self.ids[idx]
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        video_id = img_info['video_id']
        rgb_img, ir_img = self.get_image(path)
        _target = {'image_id': img_id, 'video_id': video_id, 'frame_id': img_info['frame_id']}

        # import cv2
        # import numpy as np
        # image = np.array(rgb_img)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # box = _target['annotations'][0]['bbox']
        # image = cv2.rectangle(image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color=(0, 0, 255), thickness=1)
        # cv2.imshow('rgb_img', image)

        # image = np.array(ir_img)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # box = _target['annotations'][0]['bbox']
        # image = cv2.rectangle(image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color=(0, 0, 255), thickness=1)
        # cv2.imshow('ir_img', image)
        # cv2.waitKey(0)

        rgb_img, target = self.prepare(rgb_img, _target)
        ir_img, _ = self.prepare(ir_img, _target)
        imgs.append((rgb_img, ir_img))
        if video_id == -1: # imgnet_det
            raise NotImplementedError
        else: # imgnet_vid 
            img_ids = self.cocovid.get_img_ids_from_vid(video_id)
            index = img_ids.index(img_id)
            if index >= self.num_ref_frames:
                 ref_img_ids = img_ids[index-self.num_ref_frames:index]
            else:
                if index == 0:
                    ref_img_ids = [img_ids[index]] * self.num_ref_frames
                else:
                    ref_img_ids = img_ids[:index]
                    while len(ref_img_ids) < self.num_ref_frames:
                        ref_img_ids.insert(0, ref_img_ids[0])
            ref_img_ids.sort(reverse=True)   

            for ref_img_id in ref_img_ids:
                ref_ann_ids = coco.getAnnIds(imgIds=ref_img_id)
                ref_img_info = coco.loadImgs(ref_img_id)[0]
                ref_img_path = ref_img_info['file_name']
                ref_rgb_img, ref_ir_img = self.get_image(ref_img_path)
                imgs.append((ref_rgb_img, ref_ir_img))

        if self._transforms is not None:
            rgb_imgs = [v[0] for v in imgs]
            ir_imgs = [v[1] for v in imgs]
            target_copy = target.copy()
            rgb_imgs, target = self._transforms(rgb_imgs, target) 
            ir_imgs, _ = self._transforms(ir_imgs, target_copy) 
        
        imgs = rgb_imgs + ir_imgs
        return  torch.cat(imgs, dim=0),  target

    
def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        video_id = target["video_id"]
        frame_id = target['frame_id']
        image_id = torch.tensor([image_id])
        video_id = torch.tensor([video_id])
        frame_id = torch.tensor([frame_id])
        
        target = {}

        target["image_id"] = image_id
        target["video_id"] = video_id
        target['frame_id'] = frame_id

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        
        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train_vid' or image_set == "train_det" or image_set == "train_joint":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize([600], max_size=1000),
            normalize,
        ])

    if image_set == 'test':
        return T.Compose([
            T.RandomResize([600], max_size=1000),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.vid_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "test": [(root / "Data" , root / "annotations" / 'sky_data_vid_test.json')],
    }
    datasets = []
    for (img_folder, ann_file) in PATHS[image_set]:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), is_train =(not args.eval), interval1=args.interval1,
                                interval2=args.interval2, num_ref_frames = args.num_ref_frames, return_masks=args.masks, cache_mode=args.cache_mode, 
                                local_rank=get_local_rank(), local_size=get_local_size())
        datasets.append(dataset)
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)

    

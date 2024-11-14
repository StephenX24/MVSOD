# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import json

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher_multi import data_prefetcher

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
 
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    recorder = {'video_id': [], 'frame_id': [],'loss_giou': []}
    results_store = []
    for samples, targets  in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        # image = samples.tensors.cpu()[0]
        # image = (image * torch.tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1))*255
        # image = image.permute(1, 2, 0)

        # ir_image = samples.tensors.cpu()[15]
        # ir_image = (ir_image * torch.tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1))*255
        # ir_image = ir_image.permute(1, 2, 0)
        # import numpy as np
        # import cv2 as cv
        # ir_image_np = ir_image.numpy()
        # ir_image_np = np.array(ir_image_np, dtype=np.uint8)

        # image_np = image.numpy()
        # image_np = np.array(image_np, dtype=np.uint8)

        # size = targets[0]['size'].cpu()
        # H, W = size
        # gt_boxes = targets[0]['boxes'].cpu() * torch.tensor([W, H, W, H]).reshape(1, 4)
        # gt_boxes = gt_boxes.tolist()

        # pred_logits = outputs['pred_logits'].detach().cpu()

        # value, class_index = pred_logits[0].max(1)
        # value = torch.softmax(value, 0)
        # _, box_index = value.sort(descending=True)

        # pred_boxes = outputs['pred_boxes'].detach().cpu()
        # for i in box_index[:4]:
        #     boxes = pred_boxes[0, i] * torch.tensor([W, H, W, H])
        #     v = value[i].item()
        #     if v < 0.2: continue
        #     cx, cy, w, h = boxes
        #     x1, y1, x2, y2 = int(cx-w/2),  int(cy-h/2), int(cx+w/2), int(cy+h/2)
        #     image_np = cv.rectangle(image_np.copy(), (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        #     if x1 <=10:
        #         cv.putText(image_np, f"{v:0.3f}", (x1-10, y1), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 0, 0))
        #     else:
        #         cv.putText(image_np, f"{v:0.3f}", (x1, y1), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 0, 0))

        #     ir_image_np = cv.rectangle(ir_image_np.copy(), (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        #     if x1 <=10:
        #         cv.putText(ir_image_np, f"{v:0.3f}", (x1-10, y1), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 0, 0))
        #     else:
        #         cv.putText(ir_image_np, f"{v:0.3f}", (x1, y1), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 0, 0))

        # num_obj = len(gt_boxes)
        # for box in gt_boxes:
        #     cx, cy, w, h = box
        #     x1, y1, x2, y2 = int(cx-w/2),  int(cy-h/2), int(cx+w/2), int(cy+h/2)
        #     image_np = cv.rectangle(image_np.copy(), (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        #     ir_image_np = cv.rectangle(ir_image_np.copy(), (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        # cv.putText(image_np, f"num: {num_obj}", (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), thickness=2)
        # image_np = cv.cvtColor(image_np, cv.COLOR_BGR2RGB)
        # ir_image_np = cv.cvtColor(ir_image_np, cv.COLOR_BGR2RGB)
        # im = np.concatenate([image_np, ir_image_np], 1)
        # cv.imshow('img', im)
        # import os
        # save_path = './out_figs'
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # cv.imwrite(os.path.join(save_path, f"{targets[0]['video_id'][0].item()}_{targets[0]['frame_id'][0].item()}.jpg"), im)
        # cv.waitKey(1)
        
        loss_dict = criterion(outputs, targets)
        
        video_ids = utils.all_gather(targets[0]['video_id'])
        video_ids = [v.cpu().tolist() for v in video_ids]
        frame_ids = utils.all_gather(targets[0]['frame_id'])
        frame_ids = [v.cpu().tolist() for v in frame_ids]
        loss_gious = utils.all_gather(loss_dict['loss_giou'])
        loss_gious = [v.cpu().tolist() for v in loss_gious]
        recorder['video_id'].append(video_ids)
        recorder['frame_id'].append(frame_ids)
        recorder['loss_giou'].append(loss_gious)

        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)


        # #############################################
        # image = samples.tensors.cpu()[0]
        # image = (image * torch.tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1))*255
        # image = image.permute(1, 2, 0)

        # ir_image = samples.tensors.cpu()[15]
        # ir_image = (ir_image * torch.tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1))*255
        # ir_image = ir_image.permute(1, 2, 0)
        # import numpy as np
        # import cv2 as cv
        # ir_image_np = ir_image.numpy()
        # ir_image_np = np.array(ir_image_np, dtype=np.uint8)

        # image_np = image.numpy()
        # image_np = np.array(image_np, dtype=np.uint8)
        # size = targets[0]['size'].cpu()
        # orig_size = targets[0]['orig_size'].cpu()
        # ratios = orig_size / size
        # assert ratios[0] == ratios[1]

        # orig_size = torch.flip(orig_size, dims=(0,))
        # image_np = cv.resize(image_np, orig_size.tolist(), interpolation=cv.INTER_CUBIC)
        # ir_image_np = cv.resize(ir_image_np, orig_size.tolist(), interpolation=cv.INTER_CUBIC)

        # size = targets[0]['orig_size'].cpu()
        # H, W = size
        # gt_boxes = targets[0]['boxes'].cpu() * torch.tensor([W, H, W, H]).reshape(1, 4)
        # gt_boxes = gt_boxes.tolist()

        # pred_logits = outputs['pred_logits'].detach().cpu()

        # # value, class_index = pred_logits[0].max(1)
        # # value = torch.softmax(value, 0)
        # # _, box_index = value.sort(descending=True)

        # pred_boxes = results[0]['boxes'].cpu()
        # scores = results[0]['scores'].cpu()
        # value = scores
        # # pred_boxes = outputs['pred_boxes'].detach().cpu()
        # # for i in box_index[:4]:
        # for i in range(8):
        #     # boxes = pred_boxes[0, i] * torch.tensor([W, H, W, H])
        #     boxes = pred_boxes[i]
        #     v = value[i].item()
        #     if v < 0.2: continue
        #     x1, y1, x2, y2 = boxes
        #     w = x2 - x1
        #     h = y2 - y1
        #     cx = x1 + w/2
        #     cy = y1 + h/2
        #     # cx, cy, w, h = boxes
        #     x1, y1, x2, y2 = int(cx-w/2),  int(cy-h/2), int(cx+w/2), int(cy+h/2)
        #     image_np = cv.rectangle(image_np.copy(), (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        #     if x1 <=10:
        #         cv.putText(image_np, f"{v:0.3f}", (x1-10, y1), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 0, 0))
        #     else:
        #         cv.putText(image_np, f"{v:0.3f}", (x1, y1), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 0, 0))

        #     ir_image_np = cv.rectangle(ir_image_np.copy(), (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        #     if x1 <=10:
        #         cv.putText(ir_image_np, f"{v:0.3f}", (x1-10, y1), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 0, 0))
        #     else:
        #         cv.putText(ir_image_np, f"{v:0.3f}", (x1, y1), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 0, 0))

        # num_obj = len(gt_boxes)
        # for box in gt_boxes:
        #     cx, cy, w, h = box
        #     x1, y1, x2, y2 = int(cx-w/2),  int(cy-h/2), int(cx+w/2), int(cy+h/2)
        #     image_np = cv.rectangle(image_np.copy(), (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        #     ir_image_np = cv.rectangle(ir_image_np.copy(), (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        # cv.putText(image_np, f"num: {num_obj}", (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), thickness=2)
        # image_np = cv.cvtColor(image_np, cv.COLOR_BGR2RGB)
        # ir_image_np = cv.cvtColor(ir_image_np, cv.COLOR_BGR2RGB)
        # im = np.concatenate([image_np, ir_image_np], 1)
        # cv.imshow('img', im)
        # # import os
        # # save_path = './out_figs'
        # # if not os.path.exists(save_path):
        # #     os.makedirs(save_path)
        # # cv.imwrite(os.path.join(save_path, f"{targets[0]['video_id'][0].item()}_{targets[0]['frame_id'][0].item()}.jpg"), im)
        # cv.waitKey(1)



        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        # save
        save_res = {}
        for image_id in res.keys():
            save_res[image_id] = {}
            for k, v in res[image_id].items():
                save_res[image_id][k] = v.cpu().tolist()
        results_store.append(save_res)


        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    if utils.get_rank() in [0, -1]:
        with open('eval.json', 'w') as f:
            json.dump(recorder, f)
    
    if utils.get_rank() in [0, -1]:
        with open('result_save.json', 'w') as f:
            json.dump(results_store, f)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator

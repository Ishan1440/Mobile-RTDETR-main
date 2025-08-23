"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""
'''
Implements the training and evaluation loops for the RT-DETR obj detection model
'''
import math
import os
import sys
import pathlib
from typing import Iterable

import torch
import torch.amp

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)
# from torch.profiler import profile, record_function, ProfilerActivity


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    '''
    max_norm - maximum gradient norm for gradient clipping (preventing the model from making too big changes that could make it worse instead of better) for training stability
    Function returns a dictionary containing averaged metrics for the epoch
    '''
    
    model.train()
    criterion.train() # model and loss calculator in learning mode
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(
        window_size=1, fmt='{value:.6f}')) # progress tracker to monitor how well the learning is going
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)

    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)
    show_info = True

    # with profile(
    #     # schedule=torch.profiler.schedule(
    #     #     wait=5, warmup=5, active=10, repeat=0),
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    # ) as profiler:
    # i = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device) # training images are moved to the computer's memory (GPU/CPU)

        # print(samples[0].shape)
        # print(len(samples))

        targets = [
            {k: v.to(device) for k, v in t.items()}
            for t in targets] # Takes the correct answers (targets - ground truth annotations (bounding boxes, masks)) and moves them to the same place

        # print(targets[0])

        # if show_info :
        #     summary(model,input_data=[samples,targets])
        #     show_info=False

        if scaler is not None: # Fast learning with Mixed Precision
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets) # The model looks at the images (samples) and makes predictions

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets) # The loss calculator compares predictions to correct answers

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward() #Computes gradients of the loss w.r.t. model parameters (learning signal) – this is where the model “learns from mistakes. It doesn't yet update the weights, but calculates how much each parameter should change”

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm)
            # Before applying updates, this step rescales gradients back to normal and then clips them to a maximum norm so that extremely large gradients don’t destabilize training – this is the “safety measure”

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # Actually applies the parameter update with mixed-precision scaling (scaler.step), refreshes the scaling factor (scaler.update), and clears old gradients (optimizer.zero_grad) for the next iteration.

        else:
            # Standard learning - does the same thing as above but without the speed optimizations
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)
            # updates a running avg of how well the model is performing

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"]) # Records the current learning rate

        # if i >= 50:
        #     break

        # print(i)
    #     i += 1
    # print("here")

    # profiler.export_chrome_trace("trace.json")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad() # ensures gradients aren’t tracked (saves memory & computation since we only want evaluation).
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir):
    '''
    postprocessors - Post-processing utilities for converting model outputs to detection results (bounding boxes, masks, panoptic?)
    base_ds - base dataset for COCO evaluation
    Function returns a tuple of (stats, coco_evaluator-coco eval objection with detailed results)
    Does both bounding box and segmentation evaluation
    '''
    model.eval()
    criterion.eval() # model and loss calculator in test mode

    metric_logger = MetricLogger(delimiter="  ") # to track metric across batches
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessors.iou_types # what types of IoU (Intersection over Union) to evaluate: usually "bbox" (bounding boxes), "segm" (masks).
    
    coco_evaluator = CocoEvaluator(base_ds, iou_types) # tool that implements COCO-style evaluation (AP@IoU=0.5:0.95, AP50, AR, etc.).

    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None # for panoptic segmentation
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)
        outputs = model(samples)
        #Runs the model to produce predictions - classification logits, bounding box coordinates, mask logits (depending on model).

        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)
        # postprocessors() converts raw model predictions into final format (bounding boxes in original image coordinates, masks resized properly, etc.).
        '''At this point results contain detections (eg: {boxes, scores, labels}) per image'''

        # results = postprocessors(outputs, targets)

        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {
            target['image_id'].item(): output for target,
            output in zip(targets, results)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)
        # match each image's pred with its ground truth and sent to coco_evaluator for later metric computation

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name
        #     panoptic_evaluator.update(res_pano)

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
        # compute metrics like mAP (mean Average Precision), AR (Average Recall) for bounding boxes and masks.

    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    # Extracts numeric stats from COCO evaluator into a Python dictionary.
    # Example keys:
        # "coco_eval_bbox" → [AP, AP50, AP75, AP_small, AP_medium, AP_large, …]
        # "coco_eval_masks" → same but for segmentation masks.

    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator

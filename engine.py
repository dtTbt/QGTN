# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import numpy
import torch
from torch.cuda import amp

import util.misc as utils

from util.misc import NestedTensor

from torchvision.ops import box_iou

def trs_to_nestensor(samples):
    tensors = []
    masks = []
    for sample in samples:
        tensors.append(sample['img_same_shape'])
        masks.append(sample['mask'])
    tensors = torch.stack(tensors)
    masks = torch.stack(masks)
    return NestedTensor(tensors, masks)

def box_decoder(box,device):  # (x,y,x,y) -> (xc,yc,w,h)
    boxes_t = box
    boxes = torch.zeros(boxes_t.shape,device=device)
    boxes[:, :2] = (boxes_t[:, :2] + boxes_t[:, 2:]) / 2
    boxes[:, 2:] = boxes_t[:, 2:] - boxes_t[:, :2]
    return boxes

def box_encoder(box, device):  # (xc,yc,w,h) -> (x,y,x,y)
    boxes_t = box
    boxes = torch.zeros(boxes_t.shape, device=device)
    boxes[:, :2] = boxes_t[:, :2] - boxes_t[:, 2:] / 2
    boxes[:, 2:] = boxes_t[:, :2] + boxes_t[:, 2:] / 2
    return boxes

def get_box_loss(a,b):  # (n*4,)
    x = torch.abs(a-b)
    re = x
    re = torch.sum(re)
    re = re / x.shape[0]
    return re

def get_loss(outputs, targets, device):
    boxes = outputs['boxes']

    target_boxes = []
    for target in targets:
        target_boxes.append(target['box_nml'])
    target_boxes = torch.stack(target_boxes,dim=0)
    target_boxes = box_decoder(target_boxes,device)
    target_boxes = target_boxes.unsqueeze(dim=0).repeat(boxes.shape[0],1,1)

    boxes_flat = boxes.reshape(-1)
    target_boxes_flat = target_boxes.reshape(-1)

    box_loss = get_box_loss(boxes_flat,target_boxes_flat)

    return {
        'boxes': box_loss
    }

def target_trs(targets,device):
    for t in targets:
        t['labels'] = t['labels'].to(device)
        t['boxes'] = box_decoder(t['box_nml'].unsqueeze(dim=0),device)

def get_reid_loss(a,b):
    bs, n_q = a.shape
    re = torch.sum(torch.abs(a-b))
    re = re / (bs * n_q)
    return re


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, enable_amp=None, scaler=None, auto_amp=False):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples_nes = trs_to_nestensor(samples)
        targets_nes = trs_to_nestensor(targets)
        samples_nes = samples_nes.to(device)
        targets_nes = targets_nes.to(device)

        query_boxes_list = []
        image_sizes = []
        image_sizes_gallery = []
        for sample, target in zip(samples, targets):
            box = sample['box']
            size = sample['img_same_shape'].shape[1:]
            size = tuple(size)
            box = box.unsqueeze(dim=0)
            box = box.to(device)
            query_boxes_list.append(box)
            image_sizes.append(size)
            size_gallery = target['img_same_shape'].shape[1:]
            size_gallery = tuple(size_gallery)
            image_sizes_gallery.append(size_gallery)
        if auto_amp:
            with amp.autocast(enabled=enable_amp):
                outputs = model(samples_nes, targets_nes, query_boxes_list, image_sizes, image_sizes_gallery, auto_amp=auto_amp)
        else:
            outputs = model(samples_nes, targets_nes, query_boxes_list, image_sizes, image_sizes_gallery, auto_amp=auto_amp)

        target_trs(targets, device)
        loss_dict = criterion(outputs, targets, top_matcher=False)
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
        if auto_amp:
            scaler.scale(losses).backward()
        else:
            losses.backward()

        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        if auto_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        print_loss = {
            'ce': loss_dict_reduced_scaled['loss_ce'],
            'box': loss_dict_reduced_scaled['loss_bbox'],
            'giou': loss_dict_reduced_scaled['loss_giou'],
        }
        metric_logger.update(lr=optimizer.param_groups[0]["lr"], loss=loss_value, **print_loss)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
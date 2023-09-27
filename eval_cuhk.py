import math
import os
import sys
from typing import Iterable
import time

import torch
from torch.cuda import amp

import util.misc as utils

from util.misc import NestedTensor
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
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

def get_box_loss(a,b):
    x = torch.abs(a-b)
    index_1 = torch.where(x<1)
    index_2 = torch.where(x>=1)
    re = torch.zeros_like(x)
    # re[index_1] = x[index_1]*x[index_1]*0.5
    # re[index_2] = torch.abs(x[index_2]) - 0.5
    re = x
    re = torch.sum(re)
    re = re / x.shape[0]
    return re

def post_process(outputs, targets,device):
    scores = outputs['pred_logits']
    boxes = outputs['pred_boxes']
    scores = scores.sigmoid()
    boxes_out = []
    scores_out = []
    for bs_index in range(scores.shape[0]):
        scores_tmp = scores[bs_index]
        boxes_tmp = boxes[bs_index]
        max_score = 0
        max_index = -1
        for n_index in range(scores.shape[1]):
            if scores_tmp[n_index][0] > scores_tmp[n_index][1]:
                continue
            if scores_tmp[n_index][1] > max_score:
                max_score = scores_tmp[n_index][1]
                max_index = n_index
        if max_index == -1:
            boxes_out.append(torch.tensor([-1,-1,-1,-1]).to(device))
            scores_out.append(torch.tensor(0).to(device))
        else:
            boxes_out.append(boxes_tmp[max_index])
            scores_out.append(scores_tmp[max_index][1])
    boxes_out = torch.stack(boxes_out,dim=0)  # box , (xc,yc,w,h) , (bs,4)
    scores_out = torch.stack(scores_out, dim=0)

    boxes_target = []
    for target in targets:
        boxes_target.append(target['box_nml'])
    boxes_target = torch.stack(boxes_target)  # target , (x,y,x,y) , (bs,4)

    return boxes_out, boxes_target, scores_out


def eval(model, data_loader,device,enable_amp, scaler, use_cache=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = ''
    print_freq = 100
    model.eval()
    if use_cache:
        loaded_data = torch.load('eval_cache.pth')
        all_id = loaded_data['all_id'].to(device)
        all_is = loaded_data['all_is'].to(device)
        all_box = loaded_data['all_box'].to(device)
        all_score = loaded_data['all_score'].to(device)
        all_target = loaded_data['all_target'].to(device)
    else:
        all_id = []
        all_is = []
        all_box = []
        all_score = []
        all_target = []
        for query, gallery in metric_logger.log_every(data_loader, print_freq, header):
            query_nes = trs_to_nestensor(query)
            gallery_nes = trs_to_nestensor(gallery)
            query_nes = query_nes.to(device)
            gallery_nes = gallery_nes.to(device)
            query_boxes_list = []
            image_sizes = []
            for sample in query:
                box = sample['box']
                size = sample['img_same_shape'].shape[1:]
                size = tuple(size)
                box = box.unsqueeze(dim=0)
                box = box.to(device)
                query_boxes_list.append(box)
                image_sizes.append(size)
            outputs = model(query_nes, gallery_nes, query_boxes_list, image_sizes)
            outputs, targets, scores = post_process(outputs, gallery, device)  # (bs,4) (bs,)  这里的targets仍为xyxy
            for index, ps in enumerate(query):
                all_id.append(ps['id'])
                all_is.append(not gallery[index]['id'] == -1)
            all_box.append(outputs.detach())
            all_score.append(scores.detach())
            all_target.append(targets.detach())
        all_id = torch.tensor(all_id).to(device)  # (n,)  n为eval数据集总查询对数
        all_is = torch.tensor(all_is).to(device)  # (n,)
        all_box = torch.cat(all_box,dim=0).to(device)  # (n,4)
        all_score = torch.cat(all_score, dim=0).to(device)  # (n,)
        all_target = torch.cat(all_target, dim=0).to(device)  # (n,4)
        cache_pth = './eval_cache.pth'
        if os.path.exists(cache_pth):
            os.remove(cache_pth)
        torch.save({
            'all_id': all_id,
            'all_is': all_is,
            'all_box': all_box,
            'all_score': all_score,
            'all_target': all_target
        }, cache_pth)
    num_query = torch.max(all_id).item() + 1
    num_ss = 0.
    for now_id in range(num_query):
        index = torch.where(all_id == now_id)
        boxes = all_box[index]
        scores = all_score[index]
        targets = all_target[index]
        is_one = all_is[index]
        top1_index = torch.argmax(scores)
        # while not is_one[top1_index] == True:
        #     scores[top1_index] = 0
        #     top1_index = torch.argmax(scores)
        if is_one[top1_index] == True:
            top1_box = boxes[top1_index]
            top1_box = top1_box.unsqueeze(dim=0)
            top1_box = box_encoder(top1_box,device)
            iou = box_iou(top1_box,targets[top1_index].unsqueeze(dim=0)).item()
            #print(iou)
            if iou >= 0.5:
                num_ss += 1
    top1_acc = num_ss / num_query
    print('top1_acc: ', top1_acc * 100, '%')
    return top1_acc * 100

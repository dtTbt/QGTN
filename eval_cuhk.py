import math
import os
import sys
from typing import Iterable
import time

import torch
from torch.cuda import amp

import util.misc as utils
from PIL import Image, ImageDraw, ImageFont

from util.misc import NestedTensor
from torchvision.ops import box_iou
import os
import numpy as np
from PIL import Image, ImageDraw
import shutil

from torchvision.ops import boxes as box_ops


def draw_boxes(image_path, boxes, output_folder, xyxy, sfx, keep_name, scores=None, clss=None):

    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Get image width and height
    img_width, img_height = image.size

    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
    if xyxy:
        boxes = box_decoder(boxes, boxes.device)
    # Convert boxes tensor to numpy array
    boxes_np = boxes.detach().cpu().numpy()

    for i, box in enumerate(boxes_np):
        center_x, center_y, width, height = box

        # Convert center_x, center_y, width, and height to pixel coordinates
        x1 = int((center_x - (width / 2)) * img_width)
        y1 = int((center_y - (height / 2)) * img_height)
        x2 = int((center_x + (width / 2)) * img_width)
        y2 = int((center_y + (height / 2)) * img_height)

        if scores is None:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        else:
            cls = clss[i]
            score = scores[i]

            if cls == 1:
                draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
            if cls == 2:
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            font = ImageFont.load_default()
            font_size = 20
            draw.text((x1, y1 - font_size), f"{score:.2f}", fill="black", font=font)
            draw.text((x1 + 1, y1 - font_size + 1), f"{score:.2f}", fill="white", font=font)

    # Save the image with bounding boxes
    if keep_name:
        output_filename = os.path.basename(image_path)[:-4] + '_' + sfx + os.path.basename(image_path)[-4:]
    else:
        output_filename = sfx + os.path.basename(image_path)[-4:]
    output_path = os.path.join(output_folder, output_filename)
    image.save(output_path)


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


def post_process(outputs, targets):
    scores = outputs['pred_logits']
    bs = scores.shape[0]
    scores = scores.softmax(-1)
    scores[:, :, 0] = 0.
    scores_max = scores.max(-1).values
    scores_max_index = torch.argmax(scores, dim=-1)
    keep = scores_max > 0.7

    boxes_out = []
    scores_out = []
    cls_out = []
    for bs_index in range(bs):
        keep_bs = keep[bs_index]

        boxes_out_bs = outputs['pred_boxes'][bs_index, keep_bs]
        scores_out_bs = scores_max[bs_index, keep_bs]
        cls_out_bs = scores_max_index[bs_index, keep_bs]

        boxes_out.append(boxes_out_bs)
        scores_out.append(scores_out_bs)
        cls_out.append(cls_out_bs)

    device = scores.device

    return cls_out, scores_out, boxes_out

def find_query(is_one,targets,boxes):
    true_index = torch.where(is_one == True)
    targets = targets[true_index]
    boxes = boxes[true_index]
    ious = box_iou(boxes,targets)
    right_num = 0
    all_num = boxes.shape[0]
    for i in range(all_num):
        iou = ious[i][i]
        if iou >= 0.5:
            right_num += 1
    return all_num, right_num

def eval(model, data_loader,device,enable_amp, scaler, use_cache=False, save=False, args=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = ''
    print_freq = 10
    model.eval()
    cache_path = './eval_cache.pth'
    output_folder = './look'
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    if use_cache:
        return
    else:
        for query, gallery in metric_logger.log_every(data_loader, print_freq, header):
            query_nes = trs_to_nestensor(query)
            gallery_nes = trs_to_nestensor(gallery)
            query_nes = query_nes.to(device)
            gallery_nes = gallery_nes.to(device)
            query_boxes_list = []
            image_sizes = []
            image_sizes_gallery = []
            for sample, target in zip(query, gallery):
                box = sample['boxes']
                size = sample['img_same_shape'].shape[1:]
                size = tuple(size)
                box = box.to(device)
                query_boxes_list.append(box)
                image_sizes.append(size)
                size_gallery = target['img_same_shape'].shape[1:]
                size_gallery = tuple(size_gallery)
                image_sizes_gallery.append(size_gallery)

            outputs = model(query_nes, gallery_nes, query_boxes_list, image_sizes, image_sizes_gallery)
            cls_out, scores_out, boxes_out = post_process(outputs, gallery)  # 这里的targets仍为xyxy
            # 3个都为bs个元素的list,其中cls每个元素为(n,),score每个为(n,),boxes为(n,4)

            for index, ps in enumerate(query):
                pid = int(ps['pids'])
                exist_ = gallery[index]['exist']
                if save and exist_:
                    draw_boxes(query[index]['img_path'], query[index]['boxes_nml'].squeeze(), output_folder, xyxy=True, sfx=str(pid) + '-' + str(index) + '-query', keep_name = False)
                    draw_boxes(gallery[index]['img_path'], gallery[index]['target_boxes_nml'].squeeze(), output_folder, xyxy=True, sfx=str(pid) + '-' + str(index) + '-target', keep_name = False)
                    draw_boxes(gallery[index]['img_path'], boxes_out[index], output_folder, xyxy=False, sfx=str(pid) + '-' + str(index) + '-boxes', keep_name=False, scores=scores_out[index], clss=cls_out[index])

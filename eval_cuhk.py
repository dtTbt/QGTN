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


def draw_boxes(image_path, boxes, output_folder, xyxy, sfx, keep_name, scores=None):
    """
    Draw bounding boxes on an image
    :param image_path: path to the image
    :param boxes: tensor of shape (n, 4)
    :param output_folder: folder to save the image with bounding boxes
    :param xyxy: whether the boxes are in xyxy format
    :param sfx: suffix to add to the image name
    :param keep_name: whether to keep the original image name
    :param scores: tensor of shape (n,)
    """

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

    if scores is not None:
        scores = scores.reshape(-1).detach().cpu().numpy()

    for i, box in enumerate(boxes_np):
        center_x, center_y, width, height = box

        # Convert center_x, center_y, width, and height to pixel coordinates
        x1 = int((center_x - (width / 2)) * img_width)
        y1 = int((center_y - (height / 2)) * img_height)
        x2 = int((center_x + (width / 2)) * img_width)
        y2 = int((center_y + (height / 2)) * img_height)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        if scores is not None:
            score = scores[i]
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
    scores = outputs['pred_logits']  # (bs,n_q,2)
    bs = scores.shape[0]

    scores_target = scores[..., 1]  # (bs,n_q)
    scores_target_softmax = torch.softmax(scores_target, dim=-1)  # (bs,n_q)
    max_score_index = torch.argmax(scores_target_softmax, dim=-1)  # (bs,)

    boxes_out = outputs['pred_boxes'][torch.arange(bs), max_score_index]  # (bs,4)
    scores_out = scores_target_softmax[torch.arange(bs), max_score_index]  # (bs,)

    return boxes_out, scores_out


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
        query_num_all = 0
        query_num_right = 0
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
            boxes_out, scores_out = post_process(outputs, gallery)
            # 3个都为bs个元素的list,其中cls每个元素为(n,),score每个为(n,),boxes为(n,4)

            for index, ps in enumerate(query):
                pid = int(ps['pids'])
                exist_ = gallery[index]['exist']

                # 这里暂时只考虑一张图片只出现一次query人的情况, 因此boxes_bs_num直接设定为1
                boxes_bs_num = 1
                boxes_bs_xyxy = box_encoder(boxes_out[index].unsqueeze(0), device)
                target_boxes_bs = gallery[index]['target_boxes_nml'][0].unsqueeze(0).to(device)

                query_num_all += boxes_bs_num
                ious = box_iou(boxes_bs_xyxy, target_boxes_bs)
                for i in range(boxes_bs_num):
                    if ious[i] > 0.5:
                        query_num_right += 1

                if save and exist_:
                    draw_boxes(query[index]['img_path'], query[index]['boxes_nml'], output_folder, xyxy=True, sfx=str(pid) + '-' + str(index) + '-query', keep_name = False)
                    draw_boxes(gallery[index]['img_path'], gallery[index]['target_boxes_nml'], output_folder, xyxy=True, sfx=str(pid) + '-' + str(index) + '-target', keep_name = False)
                    draw_boxes(gallery[index]['img_path'], boxes_out[index], output_folder, xyxy=False, sfx=str(pid) + '-' + str(index) + '-boxes', keep_name=False, scores=scores_out[index])

        find_query_acc = query_num_right / query_num_all
        print(f'find_query_acc: {find_query_acc * 100:.2f}%')
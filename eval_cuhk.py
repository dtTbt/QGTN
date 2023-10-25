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


def draw_boxes(image_path, boxes, output_folder, xyxy, sfx, keep_name, scores=None, is_cos_sim=False):
    """
    Draw bounding boxes on an image
    :param image_path: path to the image
    :param boxes: tensor of shape (n, 4) or (4,), after normalization
    :param output_folder: folder to save the image with bounding boxes
    :param xyxy: whether the boxes are in xyxy format. (cx,cy,w,h) when False
    :param sfx: suffix to add to the image name
    :param keep_name: whether to keep the original image name
    :param scores: tensor of shape (n,) or (,) or None
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
            if is_cos_sim and score < 0.5:
                continue
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


def draw_points(image_path, points, output_folder, sfx, keep_name, scores=None):
    """
    Draw points on an image
    :param image_path: path to the image
    :param points: tensor of shape (n, 2) or (2,), after normalization
    :param output_folder: folder to save the image with points
    :param sfx: suffix to add to the image name
    :param keep_name: whether to keep the original image name
    :param scores: tensor of shape (n,) or (,) or None
    """

    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Get image width and height
    img_width, img_height = image.size

    if points.dim() == 1:
        points = points.unsqueeze(0)

    # Convert points tensor to numpy array
    points_np = points.detach().cpu().numpy()

    if scores is not None:
        scores = scores.reshape(-1).detach().cpu().numpy()

    for i, point in enumerate(points_np):
        x, y = point

        # Convert x and y to pixel coordinates
        px = int(x * img_width)
        py = int(y * img_height)

        # Calculate point color based on score
        if scores is not None:
            # You can choose a color mapping scheme based on score value
            # For example, green for high scores and red for low scores
            score = scores[i]
            if score > 0.7:
                color = "red"
            elif score > 0.4:
                color = "yellow"
            else:
                color = "green"
        else:
            # If scores are not provided, use a default color (e.g., red)
            color = "green"

        draw.ellipse([px - 2, py - 2, px + 2, py + 2], outline=color, width=2)

    # Save the image with points
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

def feature_map_to_image_coordinates(hw, original_image_shape):
    """
    将特征图上的每个点映射到原图像上的坐标。

    参数:
    hw (tuple): 特征图的高度和宽度 (h, w)。
    original_image_shape (tuple): 原图像的形状 (height, width)。

    返回:
    torch.Tensor: 一个形状为 (h, w, 2) 的 PyTorch Tensor，
    包含特征图上每个点在原图像上的坐标。
    """
    h, w = hw
    original_image_height, original_image_width = original_image_shape

    y_ratio = original_image_height / h
    x_ratio = original_image_width / w

    # 创建坐标网格
    y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w))
    y_grid = y_grid.to(torch.float32)
    x_grid = x_grid.to(torch.float32)

    # 计算特征图上每个点在原图像上的坐标
    image_coordinates = torch.zeros((h, w, 2), dtype=torch.float32)
    image_coordinates[..., 1] = y_grid * y_ratio
    image_coordinates[..., 0] = x_grid * x_ratio

    return image_coordinates


def post_process(outputs, targets):
    pred_scores = outputs['point_scores']  # (bs,h_feat,w_feat)
    pred_boxes = outputs['pred_boxes']  # (bs,h_feat,w_feat,4)
    pred_boxes[..., :2] = - pred_boxes[...,:2]
    bs, h_feat, w_feat = pred_scores.shape
    device = pred_scores.device

    center_ref = feature_map_to_image_coordinates((h_feat, w_feat), (1, 1)).to(device)  # (feat_h,feat_w,2), (x,y)
    boxes_out = []
    scores_out = []
    points_at_img = []
    points_score = []
    for i in range(bs):
        scores_bs = pred_scores[i]  # (h_feat,w_feat)
        boxes_bs = pred_boxes[i]  # (h_feat,w_feat,4)
        boxes_bs += center_ref.repeat(1, 1, 2)  # (h_feat,w_feat,2), (xmin,ymin,xmax,ymax)
        #index = torch.where(scores_bs > 0.7)
        # 获得最大的那个点
        index = torch.argmax(scores_bs)
        index = (index // w_feat, index % w_feat)
        boxes_bs = boxes_bs[index]  # (n,4), n为置信度大于0.5的点的个数
        scores_max_bs = scores_bs[index]  # (n,)

        img_ori_shape = targets[i]['img'].shape[1:]  # (h,w)
        img_same_shape = targets[i]['img_same_shape'].shape[1:]  # (h,w)

        boxes_bs_before_nml = boxes_bs * torch.tensor(img_same_shape).flip(dims=[0]).repeat(2).reshape(1, 4).to(device)
        boxes_bs = boxes_bs_before_nml / torch.tensor(img_ori_shape).flip(dims=[0]).repeat(2).reshape(1, 4).to(device)

        boxes_out.append(boxes_bs)
        scores_out.append(scores_max_bs)

        h_img_same, w_img_same = img_same_shape
        points_at_img_bs = feature_map_to_image_coordinates((h_feat, w_feat), (h_img_same, w_img_same)).to(device)  # (feat_h,feat_w,2), (x,y)
        points_at_img_bs = points_at_img_bs / torch.tensor(img_ori_shape).flip(dims=[0]).reshape(1, 1, 2).repeat(h_feat, w_feat, 1).to(device)  # (feat_h,feat_w,2), (x,y)
        points_at_img.append(points_at_img_bs)
        points_score.append(scores_bs)

    out_dic = {
        'boxes_out': boxes_out,  # list，bs个元素，每个元素为(n,4)，n为每张图片检测出的行人数
        'scores_out': scores_out,  # list，bs个元素，每个元素为(n,)
        'points_at_img': points_at_img,  # list，bs个元素，每个元素为(feat_h,feat_w,2)，(x,y)
        'points_score': points_score,  # list，bs个元素，每个元素为(feat_h,feat_w)
    }

    return out_dic


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

            outputs = model(query_nes, gallery_nes, query_boxes_list, image_sizes, image_sizes_gallery, auto_amp=False, args=args)
            out_dic = post_process(outputs, gallery)
            boxes_out = out_dic['boxes_out']  # (bs,4)
            scores_out = out_dic['scores_out']  # (bs,)
            points_at_img = out_dic['points_at_img']  # (bs,feat_h,feat_w,2)
            points_score = out_dic['points_score']  # (bs,feat_h,feat_w)

            for index, ps in enumerate(query):
                pid = int(ps['pids'])

                #这里暂时只考虑一张图片只出现一次query人的情况, 因此boxes_bs_num直接设定为1
                boxes_bs_num = 1
                boxes_bs_xyxy = boxes_out[index]
                target_boxes_bs = gallery[index]['target_boxes_one_nml'].to(device)

                query_num_all += boxes_bs_num
                ious = box_iou(boxes_bs_xyxy, target_boxes_bs)
                flg = 0
                for i in range(boxes_bs_num):
                    if ious[i] > 0.5:
                        query_num_right += 1
                        flg = 1
                    else:
                        # 若中心点距离小于0.1，也算正确
                        center_dis = torch.sqrt(torch.sum((boxes_bs_xyxy[i, :2] - target_boxes_bs[i, :2]) ** 2))
                        if center_dis < 0.1:
                            query_num_right += 1
                            flg = 2

                if save:
                    pred_boxes_bs = boxes_out[index]
                    pred_scores_bs = scores_out[index]
                    draw_boxes(gallery[index]['img_path'], pred_boxes_bs, output_folder, xyxy=True, sfx=str(pid) + '-' + str(index) + '-pred-' + str(flg), keep_name=False, scores=pred_scores_bs)
                    # 画出target
                    target_boxes_bs = gallery[index]['target_boxes_nml']
                    draw_boxes(gallery[index]['img_path'], target_boxes_bs, output_folder, xyxy=True, sfx=str(pid) + '-' + str(index) + '-target', keep_name=False)
                    # 画出query
                    query_boxes_bs = query[index]['boxes_nml']
                    draw_boxes(query[index]['img_path'], query_boxes_bs, output_folder, xyxy=True, sfx=str(pid) + '-' + str(index) + '-query', keep_name=False)
                    # 画出points
                    points_at_img_bs = points_at_img[index]  # (feat_h,feat_w,2)
                    points_score_bs = points_score[index]  # (feat_h,feat_w)
                    points_at_img_bs = points_at_img_bs.reshape(-1, 2)  # (feat_h*feat_w,2)
                    points_score_bs = points_score_bs.reshape(-1)  # (feat_h*feat_w)
                    draw_points(gallery[index]['img_path'], points_at_img_bs, output_folder, sfx=str(pid) + '-' + str(index) + '-points', keep_name=False, scores=points_score_bs)

        find_query_acc = query_num_right / query_num_all
        print(f'find_query_acc: {find_query_acc * 100:.2f}%')

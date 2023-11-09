# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import copy
import os

from util import box_ops
from eval_cuhk import draw_points
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
from collections import OrderedDict
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import box_iou

from util.misc import feature_map_to_image_coordinates


def box_encoder(box, device=None):  # (xc,yc,w,h) -> (x,y,x,y)
    if device == None:
        device = box.device
    boxes_t = box
    boxes = torch.zeros(boxes_t.shape, device=device)
    boxes[:, :2] = boxes_t[:, :2] - boxes_t[:, 2:] / 2
    boxes[:, 2:] = boxes_t[:, :2] + boxes_t[:, 2:] / 2
    return boxes

def box_decoder(box,device):  # (x,y,x,y) -> (xc,yc,w,h)
    boxes_t = box
    boxes = torch.zeros(boxes_t.shape,device=device)
    boxes[:, :2] = (boxes_t[:, :2] + boxes_t[:, 2:]) / 2
    boxes[:, 2:] = boxes_t[:, 2:] - boxes_t[:, :2]
    return boxes

def xywhn_to_xyxy(boxes, shape):
    shape_t = shape.flip(0).repeat(2).unsqueeze(dim=0)
    shape_t = shape_t.to(boxes.device)
    boxes = box_encoder(boxes)
    boxes = boxes * shape_t
    return boxes


class ConditionalDETR(nn.Module):
    """ This is the Conditional DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, aux_loss=False, args=None):
        super().__init__()
        self.num_classes = num_classes
        self.query_feat_len = args.query_feat_len
        self.query_feat_num = pow(self.query_feat_len, 2)
        hidden_dim = transformer.d_model

        self.num_queries = args.num_queries_full
        self.query_feat_pos = nn.Embedding(self.query_feat_num, hidden_dim)
        self.score_embed = nn.Linear(hidden_dim, 1)
        self.bbox_embed = nn.Linear(hidden_dim, 4)

        #self.transformer = transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=1,
            num_decoder_layers=3,
            dim_feedforward=2048,
        )

        self.d_model = hidden_dim

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.roi_pool = MultiScaleRoIAlign(featmap_names=['feat'],
                                           output_size=(self.query_feat_len, self.query_feat_len),
                                           sampling_ratio=2
        )
        self.max_pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, samples: NestedTensor, targets, query_boxes_list, image_sizes, image_sizes_gallery, auto_amp=False, args=None,
                target_boxes_list=None,
                target_boxes_nml_s_list=None,
    ):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        bs = samples.tensors.shape[0]
        n_q = self.num_queries
        d_model = self.d_model
        device = samples.tensors.device

        # query人的特征与位置编码
        query, _ = self.backbone(samples)
        query_img_feat, _ = query[-1].decompose()
        query_img_feat = self.input_proj(query_img_feat)  # (bs,256,h,w)。缩减通道数为d_model
        query_feat = self.roi_pool(OrderedDict([["feat", query_img_feat]]), query_boxes_list, image_sizes)  # (bs,256,ql,ql)
        query_pos = self.query_feat_pos.weight  # (ql*ql,d_model)
        query_feat = query_feat.flatten(2).permute(2, 0, 1)  # (ql*ql,bs,d_model)
        query_pos = query_pos.unsqueeze(dim=1).repeat(1, bs, 1)  # (ql*ql,bs,d_model)
        query_feat_with_pos = query_feat + query_pos  # (ql*ql,bs,d_model)

        features, pos = self.backbone(targets)  # gallery
        gallery_feat, mask = features[-1].decompose()  # src即特征图(bs,c,h,w)。mask(bs,h,w)
        feat_h, feat_w = gallery_feat.shape[-2:]
        pos = pos[-1]  # 位置编码，(bs,d_model,h,w)
        gallery_feat = self.input_proj(gallery_feat)  # 通道数由2048缩减为d_model，(bs,d_model,h,w)
        gallery_feat = gallery_feat + pos  # (bs,d_model,h,w)
        gallery_feat = gallery_feat.reshape(bs, d_model, -1).permute(2, 0, 1)  # (hw,bs,d_model)
        transformer_output = self.transformer(query_feat_with_pos, gallery_feat)  # (hw,bs,d_model)

        point_scores = self.score_embed(transformer_output).sigmoid().squeeze(dim=-1)  # (hw,bs)
        point_scores = point_scores.permute(1, 0).reshape(bs, feat_h, feat_w)  # (bs,feat_h,feat_w)

        pred_boxes = self.bbox_embed(gallery_feat).sigmoid()  # (hw,bs,4)
        pred_boxes = pred_boxes.permute(1, 0, 2).reshape(bs, feat_h, feat_w, 4)  # (bs,feat_h,feat_w,4)

        out = {
            'point_scores': point_scores,  # (bs,feat_h,feat_w)
            'pred_boxes': pred_boxes,  # (bs,feat_h,feat_w,4)
        }  # 只取最后一层decoder的结果

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_points': a}
                for a in outputs_coord[:-1]]

    @torch.jit.unused
    def _set_aux_loss_for_cos(self, outputs_coord, cos_sim):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b, 'pred_logits': c}
                for b, c in zip(outputs_coord[:-1], cos_sim[:-1])]

    def _set_aux_loss_all(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]

    @torch.jit.unused
    def _set_aux_loss_split(self, outputs_class, outputs_coord, cos_sim):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'cos_sim': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], cos_sim[:-1])]

def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2, use_bce: bool = False):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    if use_bce:
        prob = inputs
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    else:
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, args=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.args = args

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        device = outputs['point_scores'].device
        pred_scores = outputs['point_scores']  # (bs,feat_h,feat_w)
        pred_boxes = outputs['pred_boxes']  # (bs,feat_h,feat_w,4)
        assert (pred_boxes >= 0).all() and (pred_boxes <= 1).all()
        tmp = torch.ones_like(pred_boxes, device=device)
        tmp[..., :2] = -1
        pred_boxes = pred_boxes * tmp  # (bs,feat_h,feat_w,4), (-x,-y,+x,+y)
        feat_h, feat_w = outputs['point_scores'].shape[-2:]
        bs = pred_scores.shape[0]

        target_boxes_nml_s_list = [t['target_boxes_one_nml_s'] for t in targets]  # 以相对same尺寸归一化的框
        target_boxes_nml_list = [t['target_boxes_one_nml'] for t in targets]  # 以相对原尺寸归一化的框
        tgt_boxes_s = torch.cat(target_boxes_nml_s_list, dim=0).to(device)  # (bs,4), (xmin,ymin,xmax,ymax)
        tgt_boxes = torch.cat(target_boxes_nml_list, dim=0).to(device)  # (bs,4), (xmin,ymin,xmax,ymax)
        assert (tgt_boxes_s[:, 2:] - tgt_boxes_s[:, :2] > 0).all()
        tgt_boxes_s_feat = tgt_boxes_s * torch.tensor([feat_w, feat_h, feat_w, feat_h], device=device).reshape(1, 4)  # (bs,4)
        mask = torch.zeros_like(pred_scores, device=device)  # (bs,feat_h,feat_w)
        ref_points = feature_map_to_image_coordinates((feat_h, feat_w), (1, 1)).to(device)  # (feat_h,feat_w,2), (x,y)
        for i in range(bs):
            index = tgt_boxes_s_feat[i].long()  # (4,)  # (xmin,ymin,xmax,ymax)
            index[-2:] += 1
            assert index[0] < index[2] and index[1] < index[3]
            mask[i, index[1]:index[3], index[0]:index[2]] = 1
            pred_boxes[i] = pred_boxes[i] + ref_points.repeat(1, 1, 2)  # (feat_h,feat_w,4), (xmin,ymin,xmax,ymax)
        tgt_boxes = tgt_boxes.unsqueeze(dim=1).repeat(1, feat_h * feat_w, 1).reshape(bs, feat_h, feat_w, 4)  # (bs,feat_h,feat_w,4)
        loss_l1 = F.smooth_l1_loss(pred_boxes, tgt_boxes, reduction='none')  # (bs,feat_h, feat_w,4)
        loss_l1 *= mask.unsqueeze(dim=-1).repeat(1, 1, 1, 4)  # (bs,feat_h,feat_w,4)
        mask_num = mask.sum()
        loss_l1 = loss_l1.sum() / mask_num

        mask_index_true = mask == 1
        pred_boxes_true = pred_boxes[mask_index_true]
        tgt_boxes_true = tgt_boxes[mask_index_true]
        loss_giou = 1 - box_ops.generalized_box_iou(pred_boxes_true, tgt_boxes_true)  # (n,n)
        loss_giou = torch.diag(loss_giou).sum() / mask_num

        loss = loss_l1 + loss_giou

        losses = {'loss_boxes': loss}
        return losses


    def loss_points(self, outputs, targets, indices, num_boxes):
        device = outputs['point_scores'].device
        pred_scores = outputs['point_scores']  # (bs,feat_h,feat_w)
        feat_h, feat_w = outputs['point_scores'].shape[-2:]
        bs = pred_scores.shape[0]
        target_boxes_nml_s_list = [t['target_boxes_one_nml_s'] for t in targets]
        tgt_boxes_s = torch.cat(target_boxes_nml_s_list, dim=0).to(device)  # (bs,4
        tgt_boxes_s = tgt_boxes_s * torch.tensor([feat_w, feat_h, feat_w, feat_h], device=device).reshape(1, 4)  # (bs,4)
        tgt_scores = torch.zeros_like(pred_scores, device=device)  # (bs,feat_h,feat_w)
        for i in range(bs):
            index = tgt_boxes_s[i].long()  # (4,)  # (xmin,ymin,xmax,ymax)
            index[-2:] += 1
            tgt_scores[i, index[1]:index[3], index[0]:index[2]] = 1
        #loss_points = sigmoid_focal_loss(pred_scores, tgt_scores, use_bce=True)  # (bs,feat_h,feat_w)
        loss_points = F.binary_cross_entropy(pred_scores, tgt_scores, reduction='none')  # (bs,feat_h,feat_w)
        loss_points = loss_points.sum() / (bs * feat_h * feat_w)
        losses = {'loss_points': loss_points}
        return losses


    def get_loss(self, loss, outputs, targets, indices=None, num_boxes=None, **kwargs):
        loss_map = {
            'points': self.loss_points,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


    def box_cos_rlt(self, losses):
        # 对loss_ce进行动态权重调整
        mt = 1 / torch.exp(losses['loss_bbox'])  # 值域为[0,1]
        mt = mt * 1
        losses['loss_ce'] = losses['loss_ce'] * mt
        for i in range(self.args.dec_layers - 1):
            mt = 1 / torch.exp(losses['loss_bbox_' + str(i)])
            mt = mt * 1
            losses['loss_ce_' + str(i)] = losses['loss_ce_' + str(i)] * mt


    def forward(self, outputs, targets, top_matcher=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        num_boxes = 1.

        # Compute all the requested losses
        losses = {}

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, num_boxes))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, num_boxes=num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = args.num_class

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args,query_feat_add=True)

    model = ConditionalDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        aux_loss=args.aux_loss,
        args=args
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_boxes': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_points'] = args.pts_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['points', 'boxes']

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses, args=args)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors

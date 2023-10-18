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

from util import box_ops
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
        hidden_dim = transformer.d_model
        if args.mode == 'cos':
            self.num_queries = args.num_queries_one
            self.query_embed = nn.Embedding(args.num_queries_one, hidden_dim)
            self.pool_to_11 = nn.AdaptiveAvgPool2d((1, 1))
        elif args.mode == 'one':
            self.num_queries = args.num_queries_one
            self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
            self.class_embed = MLP(hidden_dim, hidden_dim, num_classes, 2)
        elif args.mode == 'full':
            self.num_queries = args.num_queries_full
            self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
            self.class_embed = MLP(hidden_dim, hidden_dim, num_classes, 2)

        self.transformer = transformer
        self.d_model = hidden_dim
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.roi_pool = MultiScaleRoIAlign(featmap_names=['feat'],
                                           output_size=(self.query_feat_len, self.query_feat_len),
                                           sampling_ratio=2
        )
        # 大小需要匹配num_queries
        self.pos_pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, samples: NestedTensor, targets, query_boxes_list, image_sizes, image_sizes_gallery, auto_amp=False, args=None, target_boxes_list=None):
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

        query, pos_query = self.backbone(samples)  # query
        query_feat, _ = query[-1].decompose()
        pos_query = pos_query[-1]
        query_feat = self.input_proj(query_feat)  # (bs,256,h,w)
        query_person = self.roi_pool(OrderedDict([["feat", query_feat]]), query_boxes_list, image_sizes)  # (bs,256,ql,ql)
        if args.mode == 'cos':
            query_person_reid_feat = self.pool_to_11(query_person)  # (bs,256,1,1)
            query_person_reid_feat = query_person_reid_feat.flatten(2).permute(0, 2, 1)  # (bs,1,256)

        # query的位置编码
        pos_query = self.roi_pool(OrderedDict([["feat", pos_query]]), query_boxes_list, image_sizes)  # (bs,256,ql,ql)
        #pos_query = pos_query[:, :, 0, 0].reshape(bs, d_model, 1, 1)  # (bs,256,1,1)
        #pos_query = torch.zeros(bs,d_model,1,1).to(pos_query.device)
        #pos_query = self.pos_pool(pos)  # (bs,d_model,4,4)

        query_person = query_person.flatten(2).permute(2, 0, 1)  # (ql*ql,bs,d_model)
        pos_query = pos_query.flatten(2).permute(2, 0, 1)  # (ql*ql,bs,d_model)

        features, pos = self.backbone(targets)  # gallery
        src, mask = features[-1].decompose()  # src即特征图(bs,c,h,w)。mask(bs,h,w)
        pos = pos[-1]  # 位置编码，(bs,d_model,h,w)
        src = self.input_proj(src)  # 通道数由2048缩减为256

        dec_in = self.query_embed.weight  # (max_len,d_model)，d_model为256，为embedding的权重，即object queries

        assert mask is not None
        hs, reference = self.transformer(src, mask, dec_in, pos, query_person, pos_query, args=args)
        # hs(n_layer,bs,n_q,d_model), reference(bs,n_q,2)
        n_layer = hs.shape[0]
        hs_box = hs
        hs_score = hs

        reference_before_sigmoid = inverse_sigmoid(reference)  # 将sigmoid后的值变回去
        outputs_coord = self.bbox_embed(hs_box)  # (n_layer,bs,n_q,4)
        reference_before_sigmoid = reference_before_sigmoid.unsqueeze(dim=0).repeat(n_layer,1,1,1)
        outputs_coord[..., :2] += reference_before_sigmoid  # 给中心点加上初始量（即reference point）
        outputs_coord = outputs_coord.sigmoid()  # 框值缩放到0~1区间, (n_layer,bs,n_q,4)

        if args.test_reid:
            target_boxes_tensor = torch.stack(target_boxes_list, dim=0)  # (bs,4)
            target_boxes_tensor = box_decoder(target_boxes_tensor, device)  # (bs,4)
            outputs_coord[0,:,0,:] = target_boxes_tensor  # 将第一层的第一个预测框替换为gt框

        if args.mode == 'cos':
            # 余弦相似度
            image_sizes_gallery_single = torch.tensor(image_sizes_gallery[0])  # tensor (2,)
            outputs_coord_bs = outputs_coord.permute(1, 0, 2, 3).reshape(bs,n_layer*n_q,4)  # (bs,n_layer*n_q,4)
            boxes_list = [xywhn_to_xyxy(box, image_sizes_gallery_single) for box in outputs_coord_bs]  # bs大小的list，每个元素为tensor (n_layer*n_q,4)，为预测框映射回原图的坐标
            boxes_person = self.roi_pool(OrderedDict([["feat", src]]), boxes_list, image_sizes_gallery)  # (bs*n_layer*n_q,256,ql,ql)
            boxes_person = self.pool_to_11(boxes_person)  # (bs*n_layer*n_q,256,1,1)
            boxes_person = boxes_person.reshape(bs * n_layer, n_q, -1)  # (bs*n_layer,n_q,256)

            #boxes_person = hs.permute(1, 0, 2, 3).reshape(bs*n_layer, n_q, -1)  # (bs*n_layer,n_q,256)

            query_person_reid_feat = query_person_reid_feat / torch.norm(query_person_reid_feat, dim=2, keepdim=True)  # (bs,1,256)
            boxes_person = boxes_person / torch.norm(boxes_person, dim=2, keepdim=True)  # (bs*n_layer,n_q,256)
            query_person_reid_feat = query_person_reid_feat.unsqueeze(dim=1).repeat(1, n_layer, 1, 1).reshape(n_layer * bs, 1, -1)  # (bs*n_layer,1,256)
            boxes_with_query = torch.cat([query_person_reid_feat, boxes_person], dim=1)  # (bs*n_layer,n_q+1,256), 第0个为query
            cos_sim = torch.matmul(boxes_with_query, boxes_with_query.transpose(1, 2))  # (bs*n_layer,n_q+1,n_q+1), 值域为[-1,1]

            cos_sim_query = cos_sim.reshape(bs, n_layer, n_q + 1, n_q + 1).permute(1, 0, 2, 3)[:,:,0,1:].unsqueeze(dim=-1)  # (n_layer,bs,n_q,1)
            out = {
                'pred_logits': cos_sim_query[-1],  # (bs,n_q,1)
                'pred_boxes': outputs_coord[-1],  # (bs,n_q,4)
            }  # 只取最后一层decoder的结果
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(cos_sim_query, outputs_coord)
        else:
            outputs_class = self.class_embed(hs_score)  # (n_layer,bs,n_q,2)
            out = {
                'pred_logits': outputs_class[-1],  # (bs,n_q,2)
                'pred_boxes': outputs_coord[-1],  # (bs,n_q,4)
            }  # 只取最后一层decoder的结果
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)  # 除最后一个decoder layer的前几个层输出
                if args.fc:
                    out['pred_boxes'] = outputs_coord[args.fc_p - 1]
                    out['aux_outputs_all'] = self._set_aux_loss_all(outputs_class, outputs_coord)  # 除最后一个decoder layer的前几个层输出

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

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

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (bs,n_q,2)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o  # (bs,n_q)

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_for_cos(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'].squeeze()  # (bs,n_q)
        src_logits = src_logits * 0.5 + 0.5  # 将值域[-1,1]变为[0,1]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o  # (bs,n_q)
        target_classes = target_classes.to(torch.float32)

        loss_ce = sigmoid_focal_loss(src_logits, target_classes, num_boxes, alpha=self.focal_alpha, gamma=2, use_bce=True) * \
                    src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        return losses

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
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_cos_sim(self, outputs, targets, indices, num_boxes):
        output_cos_sim = outputs['cos_sim']  # (bs,n_q,1)
        output_boxes = outputs['pred_boxes']  # (bs,n_q,4)
        # 取得indices的cos_sim
        idx = self._get_src_permutation_idx(indices)
        cos_sim_pos = output_cos_sim[idx]  # (n_pos,1)
        # 取得除了indices的cos_sim
        num_queries = output_boxes.shape[1]
        idx_neg = self._get_src_permutation_idx_neg(indices, num_queries)
        cos_sim_neg = output_cos_sim[idx_neg]  # (n_neg,1)

        # 计算正样本的cos_sim loss，即使得cos_sim尽可能接近1
        loss_cos_sim_pos = F.l1_loss(cos_sim_pos, torch.ones_like(cos_sim_pos).to(cos_sim_pos.device))
        # 计算负样本的cos_sim loss，即使得cos_sim尽可能接近0
        loss_cos_sim_neg = F.l1_loss(cos_sim_neg, torch.zeros_like(cos_sim_neg).to(cos_sim_neg.device))
        # 计算总的cos_sim loss
        loss_cos_sim = loss_cos_sim_pos + loss_cos_sim_neg
        losses = {'loss_cos_sim': loss_cos_sim}

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_src_permutation_idx_neg(self, indices, num_queries):
        # 取得除indices外的idx
        batch_idx = []
        for i in range(len(indices)):
            # 长度为num_queries-len(indices[i][0])
            shape = num_queries - len(indices[i][0])
            batch_idx.append(torch.full((shape,), i))
        batch_idx = torch.cat(batch_idx)

        src_idx = []
        for i in range(len(indices)):
            # 长度为num_queries-len(indices[i][0])
            shape = num_queries - len(indices[i][0])
            # 从0到num_queries-1中去掉indices[i][0]中的值
            src_idx.append(torch.tensor(list(set(range(num_queries)) - set(int(x) for x in indices[i][0]))))
        src_idx = torch.cat(src_idx)
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def get_loss_for_cos(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels_for_cos,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def matcher_top(self,outputs):
        scores = outputs["pred_logits"][:,:,-1]
        tensor0 = torch.tensor([0])
        device = tensor0.device
        max_index = torch.argmax(scores,dim=1)
        re = [(index.reshape(1).to(device), tensor0) for index in max_index]
        return re


    def forward(self, outputs, targets, top_matcher=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        if self.args.fc:
            assert self.args.dec_layers > self.args.fc_p
            outputs_without_aux = {}
            outputs_without_aux['pred_boxes'] = outputs['aux_outputs_all'][self.args.fc_p - 1]['pred_boxes']
            outputs_without_aux['pred_logits'] = outputs['aux_outputs_all'][-1]['pred_logits']
        else:
            outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        if self.args.mode == 'cos':
            for loss in self.losses:
                if loss == 'masks' or loss == 'cardinality':
                    continue
                losses.update(self.get_loss_for_cos(loss, outputs, targets, indices, num_boxes))
            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    indices = self.matcher(aux_outputs, targets)
                    for loss in self.losses:
                        if loss == 'masks' or loss == 'cardinality':
                            continue
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs = {'log': False}
                        l_dict = self.get_loss_for_cos(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

            return losses

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if self.args.fc:
            assert 'aux_outputs_all' in outputs
            for i, aux_outputs in enumerate(outputs['aux_outputs_all']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    # 对于前4层，不计算label loss
                    if i + 1 <= self.args.fc_p and loss == 'labels':
                        continue
                    # 对于后2层，不计算box loss
                    if i + 1 > self.args.fc_p and loss == 'boxes':
                        continue
                    if loss == 'masks':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
            losses['loss_bbox'] = losses['loss_bbox_' + str(self.args.fc_p - 1)]
            losses['loss_giou'] = losses['loss_giou_' + str(self.args.fc_p - 1)]
            losses['loss_ce'] = losses['loss_ce_' + str(self.args.dec_layers - 1)]
            losses['cardinality_error'] = losses['cardinality_error_' + str(self.args.dec_layers - 1)]

            losses.pop('loss_bbox_' + str(self.args.fc_p - 1))
            losses.pop('loss_giou_' + str(self.args.fc_p - 1))
            losses.pop('loss_ce_' + str(self.args.dec_layers - 1))
            losses.pop('cardinality_error_' + str(self.args.dec_layers - 1))
        else:
            for loss in self.losses:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    indices = self.matcher(aux_outputs, targets)
                    for loss in self.losses:
                        if loss == 'masks':
                            continue
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs = {'log': False}
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
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
    if args.mode == 'full':
        num_classes = args.num_class
    else:
        num_classes = args.num_class - 1

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

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
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses, args=args)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
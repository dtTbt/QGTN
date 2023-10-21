# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from engine import train_one_epoch
from models import build_model

from datasets.build import build_dataset, collate_fn
from utils.transforms import build_transforms
from torch.cuda import amp

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import eval_cuhk


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr_drop', default=15, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--num_class', default=2, type=int)

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=3, type=float)
    parser.add_argument('--tgt_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    # cos_sim loss
    parser.add_argument('--cos_sim_loss_coef', default=2, type=float)

    # dataset parameters
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='/QGTN/outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--etv', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://172.0.0.1:55568', help='url used to set up distributed training')

    # QGTN parameters
    parser.add_argument('--enc_layers', default=3, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--enc_layers_2', default=3, type=int)
    parser.add_argument('--dec_layers_2', default=3, type=int)
    parser.add_argument('--num_queries_full', default=75, type=int, help='number of queries for full mode')
    parser.add_argument('--use_layer3', default=False, type=bool, help='use layer3 in backbone or not')
    parser.add_argument('--query_feat_len', default=1, type=int, help='query feature length')
    parser.add_argument('--query_self_attn', default=False, type=bool, help='query self attention or not')
    parser.add_argument('--box_cos_rlt', default=True, type=bool, help='box cos relation or not')

    # data enhancement
    parser.add_argument('--data_enhance', default=False, type=bool, help='data enhancement or not')
    parser.add_argument('--data_enhance_num', default=3, type=int, help='data enhancement number')

    # Others
    parser.add_argument('--ctn', default='', type=str)
    parser.add_argument('--eval_pth', default='/QGTN/model_epoch19.pth', type=str)
    parser.add_argument('--model_save_dir', default='./train_pth', type=str)
    parser.add_argument('--show_no_grad', default=False, type=bool)

    return parser


def resume_pth(ckpt_path, model):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt, strict=False)
    print(f"Loading pretrain pth from {ckpt_path}")


def main(args):
    global dataset_pretrain, sampler_pretrain, batch_sampler_pretrain, data_loader_pretrain
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    #dataset_train = build_dataset('CUHK-SYSU', '/CUHK-SYSU', make_coco_transforms('train'), "train", )

    dataset_train_full = build_dataset('CUHK-SYSU', '/CUHK-SYSU', build_transforms(is_train=True), "train_full", args=args)
    dataset_val = build_dataset('CUHK-SYSU', '/CUHK-SYSU', build_transforms(is_train=False), "val", args=args)
    dataset_tv = build_dataset('CUHK-SYSU', '/CUHK-SYSU', build_transforms(is_train=False), "train_val", args=args)

    if args.distributed:
        sampler_train_full = DistributedSampler(dataset_train_full)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_tv = torch.utils.data.SequentialSampler(dataset_tv)
    else:
        sampler_train_full = torch.utils.data.RandomSampler(dataset_train_full)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_tv = torch.utils.data.SequentialSampler(dataset_tv)

    batch_sampler_train_full = torch.utils.data.BatchSampler(
        sampler_train_full, args.batch_size, drop_last=True)

    data_loader_train_full = DataLoader(dataset_train_full, batch_sampler=batch_sampler_train_full,
                                   collate_fn=collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers)
    data_loader_tv = DataLoader(dataset_tv, args.batch_size, sampler=sampler_tv,
                                 drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    enable_amp = True if "cuda" in device.type else False
    scaler = amp.GradScaler(enabled=enable_amp)

    if args.eval:
        if args.eval_pth:
            resume_pth(args.eval_pth, model)
        acc_t = eval_cuhk.eval(model, data_loader_val,device,enable_amp, scaler, use_cache=False, save=True, args=args)
        exit(0)

    if args.etv:
        if args.eval_pth:
            resume_pth(args.eval_pth, model)
        acc_t = eval_cuhk.eval(model, data_loader_tv,device,enable_amp, scaler, use_cache=False, save=True, args=args)
        exit(0)

    if args.ctn:
        resume_pth(args.ctn, model_without_ddp)

    if os.path.exists(args.model_save_dir):
        shutil.rmtree(args.model_save_dir)
    os.makedirs(args.model_save_dir, exist_ok=True)

    print("Start training...")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train_full.set_epoch(epoch)

        train_one_epoch(
            model, criterion, data_loader_train_full, optimizer, device, epoch,
            args.clip_max_norm, enable_amp, scaler, auto_amp=False, args=args
        )

        lr_scheduler.step()

        if utils.is_main_process():
            save_path = os.path.join(args.model_save_dir, f'model_epoch{epoch}.pth')
            torch.save(model_without_ddp.state_dict(), save_path)

        # if (epoch + 1) % 3 == 0:
        #     acc_t = eval_cuhk.eval(model, data_loader_tv, device, enable_amp, scaler, use_cache=False, save=False, args=args)
        #     acc_t = round(acc_t, 2)
        #
        #     acc = eval_cuhk.eval(model, data_loader_val, device, enable_amp, scaler, use_cache=False, save=False, args=args)
        #     acc = round(acc, 2)
        #
        #     if utils.is_main_process():
        #         save_path = os.path.join(args.model_save_dir, f'model_epoch{epoch}_val{acc}_tv{acc_t}.pth')
        #         torch.save(model_without_ddp.state_dict(), save_path)
        #         print(f'Model saved at epoch {epoch}.')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Conditional DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

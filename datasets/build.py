import torch

from utils.transforms import build_transforms
from utils.utils import create_small_table

from .cuhk_sysu import CUHKSYSU
from .prw import PRW


def print_statistics(dataset):
    """
    Print dataset statistics.
    """
    num_imgs = len(dataset.annotations)
    num_boxes = 0
    pid_set = set()
    for anno in dataset.annotations:
        num_boxes += anno["boxes"].shape[0]
        for pid in anno["pids"]:
            pid_set.add(pid)
    statistics = {
        "dataset": dataset.name,
        "split": dataset.split,
        "num_images": num_imgs,
        "num_boxes": num_boxes,
    }
    if dataset.name != "CUHK-SYSU" or dataset.split != "query":
        pid_list = sorted(list(pid_set))
        if dataset.split == "query":
            num_pids, min_pid, max_pid = len(pid_list), min(pid_list), max(pid_list)
            statistics.update(
                {
                    "num_labeled_pids": num_pids,
                    "min_labeled_pid": int(min_pid),
                    "max_labeled_pid": int(max_pid),
                }
            )
        else:
            unlabeled_pid = pid_list[-1]
            pid_list = pid_list[:-1]  # remove unlabeled pid
            num_pids, min_pid, max_pid = len(pid_list), min(pid_list), max(pid_list)
            statistics.update(
                {
                    "num_labeled_pids": num_pids,
                    "min_labeled_pid": int(min_pid),
                    "max_labeled_pid": int(max_pid),
                    "unlabeled_pid": int(unlabeled_pid),
                }
            )
    print(f"=> {dataset.name}-{dataset.split} loaded:\n" + create_small_table(statistics))


def build_dataset(dataset_name, root, transforms, split, verbose=True, args=None):
    if dataset_name == "CUHK-SYSU":
        dataset = CUHKSYSU(root, transforms, split, args=args)
    elif dataset_name == "PRW":
        dataset = PRW(root, transforms, split)
    else:
        raise NotImplementedError(f"Unknow dataset: {dataset_name}")
    # if verbose:
    #     print_statistics(dataset)
    return dataset

from util.misc import nested_tensor_from_tensor_list

def collate_fn(batch):
    img_list = []
    for bt in batch:
        for t in bt:
            img_list.append(t['img'])
    tmp = nested_tensor_from_tensor_list(img_list)
    index = 0
    for bt in batch:
        for i, t in enumerate(bt):
            t['img_same_shape'] = tmp.tensors[index]
            t['mask'] = tmp.mask[index]
            h_same, w_same = t['img_same_shape'].shape[-2:]
            to_div = torch.tensor([w_same, h_same, w_same, h_same]).reshape(1, 4)
            if i == 0:  # query
                t['boxes_nml_s'] = t['boxes'] / to_div
            else:  # gallery
                t['boxes_nml_s'] = t['boxes'] / to_div
                t['target_boxes_nml_s'] = t['target_boxes'] / to_div
                t['target_boxes_one_nml_s'] = t['target_boxes_one'] / to_div
            index += 1
    return tuple(zip(*batch))


def build_train_loader(cfg):
    transforms = build_transforms(is_train=True)
    dataset = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "train")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.INPUT.BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=cfg.INPUT.NUM_WORKERS_TRAIN,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )


def build_test_loader(cfg):
    transforms = build_transforms(is_train=False)
    test_set = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "test")
    return torch.utils.data.DataLoader(
        test_set,
        batch_size=cfg.INPUT.BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=cfg.INPUT.NUM_WORKERS_TEST,
        pin_memory=True,
        collate_fn=collate_fn,
    )

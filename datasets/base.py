import torch
from PIL import Image
import numpy as np


class BaseDataset:
    """
    Base class of person search dataset.
    """

    def __init__(self, root, transforms, split):
        self.root = root
        self.transforms = transforms
        self.split = split
        self.annotations = self._load_annotations()

    def _load_annotations(self):
        raise NotImplementedError

    def pids_to_labels(self, pids_gallery, query_pid, boxes_gallery):
        index = np.where(pids_gallery == int(query_pid))
        index = index[0]  # ndarray (n,)
        labels = np.ones_like(pids_gallery)
        labels[index] = 2

        target_boxes = boxes_gallery[index]
        target_pids = pids_gallery[index]

        return labels, target_boxes, target_pids

    def __getitem__(self, index):
        anno = self.annotations[index]
        query, gallery = anno
        img_q = Image.open(query["img_path"]).convert("RGB")  # (w,h)
        img_g = Image.open(gallery["img_path"]).convert("RGB")
        box_q = torch.as_tensor(query["boxes"], dtype=torch.float32)
        box_g = torch.as_tensor(gallery["boxes"], dtype=torch.float32)

        labels, target_boxes, target_pids = self.pids_to_labels(gallery['pids'], query['pids'], box_g)
        labels = torch.as_tensor(labels, dtype=torch.long)

        target_pids = torch.as_tensor(target_pids, dtype=torch.long)
        target_boxes = torch.as_tensor(target_boxes, dtype=torch.float32)
        target_labels = torch.full((target_pids.shape[0],), 1, dtype=torch.long)

        A = {
            'img': img_q,
            'boxes': box_q,
            'pids': torch.as_tensor(query['pids'], dtype=torch.long),
            'img_path': query['img_path']
        }
        B = {
            'img': img_g,
            'boxes': box_g,
            'pids': torch.as_tensor(gallery['pids'], dtype=torch.long),
            'labels': labels,  # query人处为2，其他行人为1
            'img_path': gallery['img_path'],
            'exist': gallery['exist'],
            'target_boxes': target_boxes,  # (n,4)
            'target_pids': target_pids,  # (n,)
            'target_labels': target_labels
        }

        if self.transforms is not None:
            A, B = self.transforms(A, B)
        return A, B

    def __len__(self):
        return len(self.annotations)

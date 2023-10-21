import torch
from PIL import Image
import numpy as np
import random


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

    def get_image_dimensions(self, image_path):
        with Image.open(image_path) as img:
            width, height = img.size
            return height, width

    def generate_random_bbox(self, image_h, image_w, bbox_h, bbox_w):
        """
        Generate a random bounding box within the image boundaries.

        Parameters:
        image_h (int): Height of the image.
        image_w (int): Width of the image.
        bbox_h (int): Height of the bounding box.
        bbox_w (int): Width of the bounding box.

        Returns:
        tuple: Bounding box coordinates in the format (xmin, ymin, xmax, ymax).
        """
        # Ensure the bounding box dimensions are within the image dimensions
        max_x = image_w - bbox_w
        max_y = image_h - bbox_h

        # Generate random coordinates for the bounding box
        xmin = random.randint(0, max_x)
        ymin = random.randint(0, max_y)
        xmax = xmin + bbox_w
        ymax = ymin + bbox_h

        return (xmin, ymin, xmax, ymax)

    def __getitem__(self, index):
        anno = self.annotations[index]
        query, gallery, mode = anno["query"], anno["gallery"], anno["mode"]
        img_q = Image.open(query["img_path"]).convert("RGB")  # (w,h)
        img_g = Image.open(gallery["img_path"]).convert("RGB")
        box_q = torch.as_tensor(query["boxes"], dtype=torch.float32)
        box_g = torch.as_tensor(gallery["boxes"], dtype=torch.float32)

        labels, target_boxes, target_pids = self.pids_to_labels(gallery['pids'], query['pids'], box_g)
        labels = torch.as_tensor(labels, dtype=torch.long)

        target_pids = torch.as_tensor(target_pids, dtype=torch.long)
        target_boxes = torch.as_tensor(target_boxes, dtype=torch.float32)  # 未归一化，整数坐标
        target_labels = torch.full((target_pids.shape[0],), 1, dtype=torch.long)

        if mode > 0:
            img_h, img_w = self.get_image_dimensions(gallery["img_path"])  # int
            box_move = target_boxes[0]  # tensor (4,)
            box_h, box_w = int(box_move[3] - box_move[1]), int(box_move[2] - box_move[0])
            box_move_to = self.generate_random_bbox(img_h, img_w, box_h, box_w)  # (xmin, ymin, xmax, ymax)
            # 取得img_g中box_move中的内容
            box_move_src = img_g.crop(tuple(box_move.numpy()))
            # 取得img_g中box_move_to中的内容
            box_move_to_src = img_g.crop(box_move_to)
            # 将box_move_to_src覆盖img_g中box_move中的位置
            img_g.paste(box_move_to_src, tuple(box_move.numpy()))
            # 将box_move_src覆盖img_g中box_move_to中的位置
            img_g.paste(box_move_src, box_move_to)
            # 更新target_boxes为移动到的位置
            target_boxes[0] = torch.as_tensor(box_move_to, dtype=torch.float32)

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
            'labels_all_one': torch.full((labels.shape), 1, dtype=torch.long),  # 全为1
            'img_path': gallery['img_path'],
            'exist': gallery['exist'],
            'target_boxes': target_boxes,  # (n,4)
            'target_pids': target_pids,  # (n,)
            'target_labels': target_labels,  # (n,)
            'target_boxes_one': target_boxes[0].unsqueeze(dim=0),  # (1,4)
            'target_pids_one': target_pids[0].unsqueeze(dim=0),  # (1,)
            'target_labels_one': target_labels[0].unsqueeze(dim=0),  # (1,)
        }

        if self.transforms is not None:
            A, B = self.transforms(A, B)
        return A, B

    def __len__(self):
        return len(self.annotations)

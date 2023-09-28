import torch
from PIL import Image


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

    def __getitem__(self, index):
        anno = self.annotations[index]
        img_q = Image.open(anno[0]["img_path"]).convert("RGB")  # (w,h)
        img_g = Image.open(anno[1]["img_path"]).convert("RGB")
        box_q = torch.as_tensor(anno[0]["box"], dtype=torch.float32)
        box_g = torch.as_tensor(anno[1]["box"], dtype=torch.float32)
        img = {'img': img_q,
               'box': box_q,
               'id': anno[0]['id'],
               'img_path': anno[0]['img_path']
               }
        target = {'img': img_g,
                  'box': box_g,
                  'is_one': anno[2],
                  'id': anno[1]['id'],
                  'labels': torch.tensor([1]),
                  'img_path': anno[1]['img_path']
                  }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.annotations)

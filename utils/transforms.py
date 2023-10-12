import random

from torchvision.transforms import functional as F
import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class ToTensor:
    def __call__(self, image, target):
        # convert [0, 255] to [0, 1]
        image['img'] = F.to_tensor(image['img'])
        target['img'] = F.to_tensor(target['img'])

        image['boxes_nml'] = image['boxes'] / (torch.flip(torch.tensor(image['img'].shape[1:]),dims=[0])).repeat(2).reshape(1,4)
        target['boxes_nml'] = target['boxes'] / (torch.flip(torch.tensor(target['img'].shape[1:]), dims=[0])).repeat(2).reshape(1,4)
        target['target_boxes_nml'] = target['target_boxes'] / (torch.flip(torch.tensor(target['img'].shape[1:]), dims=[0])).repeat(2).reshape(1, 4)

        assert (image['boxes_nml'][:, 2:] >= image['boxes_nml'][:, :2]).all()
        assert (target['boxes_nml'][:, 2:] >= target['boxes_nml'][:, :2]).all()
        assert (target['target_boxes_nml'][:, 2:] >= target['target_boxes_nml'][:, :2]).all()

        return image, target


def build_transforms(is_train):
    transforms = []
    transforms.append(ToTensor())
    # if is_train:
    #     transforms.append(RandomHorizontalFlip())
    return Compose(transforms)

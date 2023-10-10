import os.path
import os.path as osp

import numpy as np
import torch
from scipy.io import loadmat
from PIL import Image

from .base import BaseDataset


class CUHKSYSU(BaseDataset):
    def __init__(self, root, transforms, split):
        self.name = "CUHK-SYSU"
        self.img_prefix = osp.join(root, "Image", "SSM")
        super(CUHKSYSU, self).__init__(root, transforms, split)

    def get_train_img_name(self):  # 返回需要的图片名
        gallery_imgs = loadmat(osp.join(self.root, "annotation", "pool.mat"))
        gallery_imgs = gallery_imgs["pool"].squeeze()
        gallery_imgs = [str(a[0]) for a in gallery_imgs]
        all_imgs = loadmat(osp.join(self.root, "annotation", "Images.mat"))
        all_imgs = all_imgs["Img"].squeeze()
        all_imgs = [str(a[0][0]) for a in all_imgs]
        training_imgs = sorted(list(set(all_imgs) - set(gallery_imgs)))
        return training_imgs

    def get_train_data_index(self, n):
        index = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                index.append([i,j,1])
        return index

    def get_test_data_index(self,n):
        index = []
        for i in range(n)[:-1]:
            index.append([0,i+1,1])
        return index

    def get_image_dimensions(self, image_path):
        with Image.open(image_path) as img:
            width, height = img.size
            return height, width

    def generate_random_boxes(self, image_height, image_width, num_boxes):
        boxes = np.zeros((num_boxes, 4), dtype=np.int32)

        for i in range(num_boxes):
            x1 = np.random.randint(0, image_width)
            y1 = np.random.randint(0, image_height)
            x2 = np.random.randint(x1, image_width)
            y2 = np.random.randint(y1, image_height)

            boxes[i] = [x1, y1, x2, y2]

        return boxes

    def _load_annotations(self):
        if self.split == "train":
            train_data = []
            train = loadmat(osp.join(self.root, "annotation/test/train_test/Train.mat"))
            train = train["Train"].squeeze()
            for index, item in enumerate(train):  # 枚举每个人，从0开始编号
                scenes = item[0, 0][2].squeeze()
                train_gallery = []
                for img_name, box, _ in scenes:  # 枚举这个人的每一次出现（出现在的图片与图片中的位置）
                    img_name = str(img_name[0])
                    box = box.squeeze().astype(np.int32)
                    box[2:] += box[:2]
                    assert (box[2:] >= box[:2]).all()
                    train_gallery.append({
                        'id': index,
                        'img_name': img_name,
                        'box': box,
                        'img_path': os.path.join(self.img_prefix, img_name),
                    })
                train_data_indexes=self.get_train_data_index(len(train_gallery))
                for train_data_index in train_data_indexes:
                    index_l, index_r, is_one = train_data_index
                    train_data.append([train_gallery[index_l],train_gallery[index_r],is_one])
            return train_data
        elif self.split == "val":
            test_data = []
            protoc = loadmat(osp.join(self.root, "annotation/test/train_test/TestG50.mat"))
            protoc = protoc["TestG50"].squeeze()
            for index, item in enumerate(protoc):  # 枚举每一个query人
                # query
                im_name = str(item["Query"][0, 0][0][0])
                box = item["Query"][0, 0][1].squeeze().astype(np.int32)
                box[2:] += box[:2]
                test_gallery = []
                test_gallery.append({
                    'id': index,
                    'img_name': im_name,
                    'box': box,
                    'img_path': os.path.join(self.img_prefix, im_name)
                })
                # gallery
                gallery = item["Gallery"].squeeze()
                for im_name, box, _ in gallery:  # 枚举gallery中每张图片（50张）
                    im_name = str(im_name[0])
                    box = box.squeeze().astype(np.int32)
                    box[2:] += box[:2]
                    idd = index
                    if len(box) == 0:
                        box = np.array([1,1,1,1])
                        idd = -1
                    test_gallery.append({
                        'id': idd,
                        'img_name': im_name,
                        'box': box,
                        'img_path': os.path.join(self.img_prefix, im_name)
                    })
                test_data_indexes=self.get_test_data_index(len(test_gallery))
                for test_data_index in test_data_indexes:
                    index_l, index_r, is_one = test_data_index
                    test_data.append([test_gallery[index_l],test_gallery[index_r],is_one])
            return test_data
        elif self.split == "val100":
            test_data = []
            protoc = loadmat(osp.join(self.root, "annotation/test/train_test/TestG50.mat"))
            protoc = protoc["TestG50"].squeeze()
            for index, item in enumerate(protoc):  # 枚举每一个query人
                # query
                if index >= 200:
                    break
                im_name = str(item["Query"][0, 0][0][0])
                box = item["Query"][0, 0][1].squeeze().astype(np.int32)
                box[2:] += box[:2]
                test_gallery = []
                test_gallery.append({
                    'id': index,
                    'img_name': im_name,
                    'box': box,
                    'img_path': os.path.join(self.img_prefix, im_name)
                })
                # gallery
                gallery = item["Gallery"].squeeze()
                for im_name, box, _ in gallery:  # 枚举gallery中每张图片（50张）
                    im_name = str(im_name[0])
                    box = box.squeeze().astype(np.int32)
                    box[2:] += box[:2]
                    idd = index
                    if len(box) == 0:
                        box = np.array([1,1,1,1])
                        idd = -1
                        break
                    test_gallery.append({
                        'id': idd,
                        'img_name': im_name,
                        'box': box,
                        'img_path': os.path.join(self.img_prefix, im_name)
                    })
                test_data_indexes=self.get_test_data_index(len(test_gallery))
                for test_data_index in test_data_indexes:
                    index_l, index_r, is_one = test_data_index
                    test_data.append([test_gallery[index_l],test_gallery[index_r],is_one])
            return test_data
        elif self.split == 'tv100':
            train_data = []
            train = loadmat(osp.join(self.root, "annotation/test/train_test/Train.mat"))
            train = train["Train"].squeeze()
            for index, item in enumerate(train):  # 枚举每个人，从0开始编号
                if index >= 100:
                    break
                scenes = item[0, 0][2].squeeze()
                train_gallery = []
                for img_name, box, _ in scenes:  # 枚举这个人的每一次出现（出现在的图片与图片中的位置）
                    img_name = str(img_name[0])
                    box = box.squeeze().astype(np.int32)
                    box[2:] += box[:2]
                    assert (box[2:] >= box[:2]).all()
                    train_gallery.append({
                        'id': index,
                        'img_name': img_name,
                        'box': box,
                        'img_path': os.path.join(self.img_prefix, img_name),
                    })
                train_data_indexes=self.get_train_data_index(len(train_gallery))
                for train_data_index in train_data_indexes:
                    index_l, index_r, is_one = train_data_index
                    train_data.append([train_gallery[index_l],train_gallery[index_r],is_one])
            return train_data
        elif self.split == "pretrain":
            img_names = self.get_train_img_name()
            train_data = []
            for index, img_name in enumerate(img_names):  # 枚举每个人，从0开始编号
                img_path = os.path.join(self.img_prefix, img_name)
                h, w = self.get_image_dimensions(img_path)
                boxes = self.generate_random_boxes(h, w, 10)
                for box in boxes:
                    box = box.astype(np.int32)
                    tmp = {
                        'id': -2,
                        'img_name': img_name,
                        'box': box,
                        'img_path': img_path
                    }
                    train_data.append([tmp,tmp,1])
            return train_data

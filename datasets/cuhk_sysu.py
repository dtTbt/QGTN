import os.path
import os.path as osp

import numpy as np
import torch
from scipy.io import loadmat

from .base import BaseDataset


class CUHKSYSU(BaseDataset):
    def __init__(self, root, transforms, split):
        self.name = "CUHK-SYSU"
        self.img_prefix = osp.join(root, "Image", "SSM")
        super(CUHKSYSU, self).__init__(root, transforms, split)

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
                # if index >= 100:
                #     break
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

import os.path
import os.path as osp

import numpy as np
import torch
from scipy.io import loadmat
from PIL import Image

from .base import BaseDataset


class CUHKSYSU(BaseDataset):
    def __init__(self, root, transforms, split, args=None):
        self.name = "CUHK-SYSU"
        self.img_prefix = osp.join(root, "Image", "SSM")
        self.args = args
        super(CUHKSYSU, self).__init__(root, transforms, split)

    def get_img_name(self):  # 返回需要的图片名
        gallery_imgs = loadmat(osp.join(self.root, "annotation", "pool.mat"))
        gallery_imgs = gallery_imgs["pool"].squeeze()
        gallery_imgs = [str(a[0]) for a in gallery_imgs]
        if self.split == "val":
            return gallery_imgs

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
                index.append([i, j])
        return index

    def get_test_data_index(self, n):
        index = []
        for i in range(n)[:-1]:
            index.append([0, i + 1])
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

    def load_img(self):
        def set_box_pid(boxes, box, pids, pid):  # 在一张图片的boxes中找到匹配的box，然后赋值pid
            for i in range(boxes.shape[0]):
                if np.all(boxes[i] == box):
                    pids[i] = pid
                    return

        all_imgs = loadmat(osp.join(self.root, "annotation", "Images.mat"))
        all_imgs = all_imgs["Img"].squeeze()
        name_to_boxes = {}
        name_to_pids = {}
        unlabeled_pid = 5555  # default pid for unlabeled people
        for img_name, _, boxes in all_imgs:  # 枚举每张图片
            img_name = str(img_name[0])
            boxes = np.asarray([b[0] for b in boxes[0]])
            boxes = boxes.reshape(boxes.shape[0], 4)  # (x1, y1, w, h)
            valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
            assert valid_index.size > 0, "Warning: {} has no valid boxes.".format(img_name)
            boxes = boxes[valid_index]
            name_to_boxes[img_name] = boxes.astype(np.int32)  # 每张图片的所有框
            name_to_pids[img_name] = unlabeled_pid * np.ones(boxes.shape[0], dtype=np.int32)  # 所有框id全部赋为5555

        if self.split == 'val':
            protoc = loadmat(osp.join(self.root, "annotation/test/train_test/TestG50.mat"))
            protoc = protoc["TestG50"].squeeze()
            for index, item in enumerate(protoc):  # 枚举每一个query人
                im_name = str(item["Query"][0, 0][0][0])
                box = item["Query"][0, 0][1].squeeze().astype(np.int32)
                set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index + 1)  # 为查询框编号
                gallery = item["Gallery"].squeeze()
                for im_name, box, _ in gallery:  # 枚举gallery中每张图片（50张）
                    im_name = str(im_name[0])
                    if box.size == 0:  # 因为gallery中只有前几张图片是存在这个人的，后面的没有这人
                        break
                    box = box.squeeze().astype(np.int32)
                    set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index + 1)  # 为gallery中这人框编号
        else:
            train = loadmat(osp.join(self.root, "annotation/test/train_test/Train.mat"))
            train = train["Train"].squeeze()
            for index, item in enumerate(train):  # 枚举每个人，从0开始编号
                scenes = item[0, 0][2].squeeze()
                for img_name, box, _ in scenes:  # 枚举这个人的每一次出现（出现在的图片与图片中的位置）
                    img_name = str(img_name[0])
                    box = box.squeeze().astype(np.int32)
                    set_box_pid(name_to_boxes[img_name], box, name_to_pids[img_name], index + 1)  # 为每一次出现的框都编号

        annotations = {}
        img_names = self.get_img_name()  # 获取需要的图片名,val or train
        max_person_num = 0
        for img_name in img_names:
            boxes = name_to_boxes[img_name]
            boxes[:, 2:] += boxes[:, :2]  # (x1, y1, w, h) -> (x1, y1, x2, y2)
            pids = name_to_pids[img_name]
            annotations[img_name] = {
                "img_name": img_name,
                "img_path": osp.join(self.img_prefix, img_name),
                "boxes": boxes,
                "pids": pids,
            }
            max_person_num = max(max_person_num, len(pids))
        return annotations, max_person_num

    def _load_annotations(self):
        imgs_set, _ = self.load_img()
        if self.split == "train_full":
            boxes_num = {'0 people': 0, '1 people': 0, '2 people': 0, 'more than 2': 0}
            train_data = []
            train = loadmat(osp.join(self.root, "annotation/test/train_test/Train.mat"))
            train = train["Train"].squeeze()
            for index, item in enumerate(train):  # 枚举每个人，从0开始编号
                pid = index + 1
                scenes = item[0, 0][2].squeeze()
                train_gallery = []
                train_query = []
                for img_name, box, _ in scenes:  # 枚举这个人的每一次出现（出现在的图片与图片中的位置）
                    img_name = str(img_name[0])

                    box = box.squeeze().astype(np.int32)
                    box[2:] += box[:2]
                    box = box.reshape(1, 4)
                    train_query.append({
                        'img_name': img_name,
                        'pids': np.array([pid]),  # ndarray (1,)
                        'boxes': box,  # ndarray (1,4)
                        'img_path': os.path.join(self.img_prefix, img_name),
                    })

                    gallery_img = imgs_set[img_name]
                    boxes = gallery_img['boxes']
                    pids = gallery_img['pids']
                    train_gallery.append({
                        'img_name': img_name,
                        'pids': pids,  # ndarray (n,)
                        'boxes': boxes,  # ndarray (n,4)
                        'img_path': os.path.join(self.img_prefix, img_name),
                        'exist': True,
                    })
                    # 查找pids中有几个为pid的
                    pids_num = np.sum(pids == pid)
                    if pids_num <= 2:
                        boxes_num[str(pids_num) + ' people'] += 1
                    else:
                        boxes_num['more than 2'] += 1
                train_data_indexes = self.get_train_data_index(len(scenes))
                for train_data_index in train_data_indexes:
                    index_l, index_r = train_data_index
                    train_data.append({
                        'query': train_query[index_l],
                        'gallery': train_gallery[index_r],
                        'mode': 0  # 代表使用原图片
                    })
                if self.args.data_enhance:
                    for i in range(self.args.data_enhance_num):
                        enhance_index = i + 1
                        for train_data_index in train_data_indexes:
                            index_l, index_r = train_data_index
                            train_data.append({
                                'query': train_query[index_l],
                                'gallery': train_gallery[index_r],
                                'mode': enhance_index
                            })
            print(boxes_num)
            return train_data
        elif self.split == "val":
            test_data = []
            protoc = loadmat(osp.join(self.root, "annotation/test/train_test/TestG50.mat"))
            protoc = protoc["TestG50"].squeeze()
            for index, item in enumerate(protoc):  # 枚举每一个query人
                pid = index + 1
                if index >= 100:  #
                    break
                im_name = str(item["Query"][0, 0][0][0])
                box = item["Query"][0, 0][1].squeeze().astype(np.int32)
                box[2:] += box[:2]
                box = box.reshape(1, 4)
                test_gallery = []
                test_gallery.append({
                    'pids': np.array([pid]),
                    'img_name': im_name,
                    'boxes': box,
                    'img_path': os.path.join(self.img_prefix, im_name)
                })
                # gallery
                gallery = item["Gallery"].squeeze()
                for im_name, box, _ in gallery:  # 枚举gallery中每张图片（50张）
                    img_name = str(im_name[0])
                    if box.shape[-1] == 0:  # 因为gallery中只有前几张图片是存在这个人的，后面的没有这人
                        break
                    gallery_img = imgs_set[img_name]
                    boxes = gallery_img['boxes']
                    pids = gallery_img['pids']
                    test_gallery.append({
                        'img_name': img_name,
                        'pids': pids,  # ndarray (n,)
                        'boxes': boxes,  # ndarray (n,4)
                        'img_path': os.path.join(self.img_prefix, img_name),
                        'exist': True,
                    })
                test_data_indexes = self.get_test_data_index(len(test_gallery))
                for test_data_index in test_data_indexes:
                    index_l, index_r = test_data_index
                    test_data.append({
                        'query': test_gallery[index_l],
                        'gallery': test_gallery[index_r],
                        'mode': 0  # 代表使用原图片
                    })
            return test_data
        elif self.split == "train_val":
            train_data = []
            train = loadmat(osp.join(self.root, "annotation/test/train_test/Train.mat"))
            train = train["Train"].squeeze()
            for index, item in enumerate(train):  # 枚举每个人，从0开始编号
                if index >= 100:  #
                    break
                pid = index + 1
                scenes = item[0, 0][2].squeeze()
                train_gallery = []
                train_query = []
                for img_name, box, _ in scenes:  # 枚举这个人的每一次出现（出现在的图片与图片中的位置）
                    img_name = str(img_name[0])

                    box = box.squeeze().astype(np.int32)
                    box[2:] += box[:2]
                    box = box.reshape(1, 4)
                    train_query.append({
                        'img_name': img_name,
                        'pids': np.array([pid]),  # ndarray (1,)
                        'boxes': box,  # ndarray (1,4)
                        'img_path': os.path.join(self.img_prefix, img_name),
                    })

                    gallery_img = imgs_set[img_name]
                    boxes = gallery_img['boxes']
                    pids = gallery_img['pids']
                    train_gallery.append({
                        'img_name': img_name,
                        'pids': pids,  # ndarray (n,)
                        'boxes': boxes,  # ndarray (n,4)
                        'img_path': os.path.join(self.img_prefix, img_name),
                        'exist': True,
                    })
                train_data_indexes = self.get_test_data_index(len(train_query))
                for train_data_index in train_data_indexes:
                    index_l, index_r = train_data_index
                    train_data.append({
                        'query': train_query[index_l],
                        'gallery': train_gallery[index_r],
                        'mode': 0  # 代表使用原图片
                    })
            return train_data

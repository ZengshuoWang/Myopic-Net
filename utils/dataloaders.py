import random
import sys
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import os
from glob import glob
from datetime import datetime

from .transforms import TrainImgPairsAugmentor


class MdcpsDataset(data.Dataset):
    def __init__(self, root=None, usage=None, perform_aug=False, aug_params_tf1=None):
        """
        root: str, 数据集的路径
        usage: str, "train" or "eval" or "train_eval"
        perform_aug: bool, True or False, 对于输入的原始图像对是否进行简单的数据增强
        aug_params: dict, 数据增强操作所需要的参数
        """
        assert (root is not None)
        assert ((usage == "train") or (usage == "eval") or (usage == "train_eval"))

        self.usage = usage
        if usage == "train_eval":
            usage = "train"
        self.perform_aug = perform_aug
        self.augmentor_tf1 = TrainImgPairsAugmentor(**aug_params_tf1)

        root_md_gt = os.path.join(root, usage, "myopia_development_gt.txt")
        self.md_gt_list = open(root_md_gt, "r", encoding='utf-8').read().splitlines()

        root_img_pairs = os.path.join(root, usage, "img_pairs")
        img_pairs = sorted(glob(os.path.join(root_img_pairs, '*.jpg')))
        self.img_pairs_list = []
        for i in range(len(img_pairs) // 2):
            self.img_pairs_list += [[img_pairs[2 * i], img_pairs[2 * i + 1]]]

        self.init_seed = False

    def __getitem__(self, index):
        if not self.init_seed:
            worker_infor = torch.utils.data.get_worker_info()
            if worker_infor is not None:
                torch.manual_seed(worker_infor.id)
                np.random.seed(worker_infor.id)
                random.seed(worker_infor.id)
                self.init_seed = True

        if self.usage == "eval":
            img1 = Image.open(self.img_pairs_list[index][0])
            img2 = Image.open(self.img_pairs_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

            md_gt = int(self.md_gt_list[index])

            return img1, img2, md_gt

        elif self.usage == "train_eval":
            img1 = Image.open(self.img_pairs_list[index][0])
            img2 = Image.open(self.img_pairs_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

            md_gt = int(self.md_gt_list[index])

            return img1, img2, md_gt

        elif self.usage == 'train':
            index = index % len(self.img_pairs_list)

            img1 = Image.open(self.img_pairs_list[index][0])
            img2 = Image.open(self.img_pairs_list[index][1])

            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)

            img1 = img1[..., :3]
            img2 = img2[..., :3]

            if self.perform_aug:
                img1, img2 = self.augmentor_tf1(img1, img2)

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

            md_gt = int(self.md_gt_list[index])

            return img1, img2, md_gt

        else:
            assert ((self.usage == "train") or (self.usage == "eval") or (self.usage == "train_eval"))

    def __len__(self):
        return len(self.img_pairs_list)


class TestDataset(data.Dataset):
    def __init__(self, root=None):
        assert (root is not None)

        root_md_gt = os.path.join(root, "myopia_development_gt.txt")
        self.md_gt_list = open(root_md_gt, "r", encoding='utf-8').read().splitlines()

        root_img_pairs = os.path.join(root, "img_pairs")
        img_pairs = sorted(glob(os.path.join(root_img_pairs, '*.jpg')))
        self.img_pairs_list = []
        for i in range(len(img_pairs) // 2):
            self.img_pairs_list += [[img_pairs[2 * i], img_pairs[2 * i + 1]]]

    def __getitem__(self, index):
        img1_img2_names = str(self.img_pairs_list[index][0].split("/")[-1] + "&" +
                              self.img_pairs_list[index][1].split("/")[-1])

        img1 = Image.open(self.img_pairs_list[index][0])
        img2 = Image.open(self.img_pairs_list[index][1])
        img1 = np.array(img1).astype(np.uint8)[..., :3]
        img2 = np.array(img2).astype(np.uint8)[..., :3]
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        md_gt = int(self.md_gt_list[index])

        return img1, img2, md_gt, img1_img2_names

    def __len__(self):
        return len(self.img_pairs_list)


def create_dataloaders(config, shuffle_train=True, shuffle_eval=False, shuffle_train_eval=False):
    """ Create the data loader for the network """
    # shuffle_train: 表示在训练的时候加载数据是否打乱顺序，理想的情况下当然是打乱顺序，但是这里要考虑到一点：
    # 这里使用了多个dataloader，也就是说在训练的时候会有多组相似的输入图像组img1, img2，
    # 如果不打乱的话，这多组相似的图像基本挨着，但是打乱后就不会挨着了，就比较分散了。

    aug_params_tf1 = {'do_flip': True}

    dataloaders_train = []
    dataset_train = MdcpsDataset(root=config.dataset_root, usage="train", perform_aug=True,
                                 aug_params_tf1=aug_params_tf1)
    for d_i in range(config.num_dataloaders):
        print("Creating dataloader for train index %d out of %d time %s" %
              (d_i, config.num_dataloaders, datetime.now()))
        sys.stdout.flush()
        dataloader_train = data.DataLoader(dataset_train, batch_size=int(config.dataloader_batch_sz),
                                           shuffle=shuffle_train, num_workers=40, drop_last=False)
        dataloaders_train.append(dataloader_train)

    dataset_eval = MdcpsDataset(root=config.dataset_root, usage="eval",
                                perform_aug=False, aug_params_tf1=aug_params_tf1)
    dataloader_eval = data.DataLoader(dataset_eval, batch_size=config.batch_sz,
                                      shuffle=shuffle_eval, num_workers=40, drop_last=False)

    dataset_train_eval = MdcpsDataset(root=config.dataset_root, usage="train_eval",
                                      perform_aug=False, aug_params_tf1=aug_params_tf1)
    dataloader_train_eval = data.DataLoader(dataset_train_eval, batch_size=config.batch_sz,
                                            shuffle=shuffle_train_eval, num_workers=40, drop_last=False)

    return dataloaders_train, dataloader_eval, dataloader_train_eval


def create_dataloader_test(config, shuffle=False):
    dataset_test = TestDataset(root=config.dataset_root)
    dataloader_test = data.DataLoader(dataset_test, batch_size=config.batch_sz,
                                      shuffle=shuffle, num_workers=40, drop_last=False)

    return dataloader_test

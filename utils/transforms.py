import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torchvision.transforms as tf
import torchvision
from PIL import Image


def sobel_process(imgs, include_rgb=True):
    bn, c, h, w = imgs.size()

    assert (c == 3)  # rgb图像

    # Grayscale()的输入图像可以是PIL图像或Tensor，在这种情况下，它应该具有[..., 3, h, w]的形状，
    # 其中，...表示任意数量的前导维数
    transform = tf.Compose(
        [
            tf.Grayscale(num_output_channels=1)
        ]
    )
    grey_imgs = transform(imgs)  # bn, 1, h, w
    assert (grey_imgs.shape == (bn, 1, h, w))

    sobel1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight = nn.Parameter(torch.Tensor(sobel1).cuda().float().unsqueeze(0).unsqueeze(0))
    dx = conv1(Variable(grey_imgs)).data

    sobel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight = nn.Parameter(torch.from_numpy(sobel2).cuda().float().unsqueeze(0).unsqueeze(0))
    dy = conv2(Variable(grey_imgs)).data

    sobel_imgs = torch.cat([dx, dy], dim=1)
    assert (sobel_imgs.shape == (bn, 2, h, w))

    if include_rgb:
        rgb_sobel_imgs = torch.cat([imgs, sobel_imgs], dim=1)
        assert (rgb_sobel_imgs.shape == (bn, 5, h, w))

        return rgb_sobel_imgs

    else:
        return sobel_imgs


def gray_process(imgs):
    bn, c, h, w = imgs.size()

    assert (c == 3)

    transform = tf.Compose(
        [
            tf.Grayscale(num_output_channels=1)
        ]
    )
    gray_imgs = transform(imgs)  # bn, 1, h, w
    assert (gray_imgs.shape == (bn, 1, h, w))

    return gray_imgs


# tf1 ---> 对原始的train img pairs进行数据增强的操作
class TrainImgPairsAugmentor:
    def __init__(self, do_flip=True):
        # 翻转增强参数
        self.do_flip = do_flip  # 进行翻转
        self.h_flip_prob = 0.5  # 水平翻转的概率
        self.v_flip_prob = 0.5  # 垂直翻转的概率

        # 光度增强参数
        self.photo_aug = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                                            saturation=0.4, hue=0.2)
        self.photo_aug_prob = 0.5  # 进行光度增强的概率
        self.asymmetric_color_aug_prob = 0.5  # 不对称颜色增强的概率

    # 光度变换函数
    def color_transform(self, img1, img2):
        if np.random.rand() < self.photo_aug_prob:  # 以一定的概率进行光度变换
            # 不对称变换，就是图像对中的两幅图像各自变换各自的
            if np.random.rand() < self.asymmetric_color_aug_prob:
                # 首先使用Image.fromarray()方法把image实现从array到Image的变换
                # 然后调用ColorJitter()函数对图像进行颜色空间的变换
                img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
                img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

            # 对称变换
            else:
                img_pair_stack = np.concatenate([img1, img2], axis=0)  # h, w, c
                img_pair_stack = np.array(self.photo_aug(Image.fromarray(img_pair_stack)), dtype=np.uint8)
                img1, img2 = np.split(img_pair_stack, 2, axis=0)  # 分离两幅图像

        return img1, img2

    # 随机空间变换
    def spatial_transform(self, img1, img2):
        # 以一定的概率进行翻转变换
        if self.do_flip:
            # 以0.5的概率进行水平翻转
            if np.random.rand() < self.h_flip_prob:
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
            # 以0.5的概率进行垂直翻转
            if np.random.rand() < self.v_flip_prob:
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
        return img1, img2

    def __call__(self, img1, img2):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.spatial_transform(img1, img2)

        # 将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度快
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)

        return img1, img2

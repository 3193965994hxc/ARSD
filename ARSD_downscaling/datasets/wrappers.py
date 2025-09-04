import random
import math
from scipy.ndimage import zoom
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import register
from utils import to_pixel_samples
from  imresize import imresize

def bicubic(img, size=(48, 48)):
    img=img[:,:,np.newaxis]
    lr = imresize(img,  output_shape=size, kernel='cubic', channel=1)
    lr = lr[:, :, 0]
    return np.array(lr).astype(np.float32)

@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        # 初始化数据集参数
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        # 返回数据集的大小
        return len(self.dataset)

    def __getitem__(self, idx):

        # 获取原始图像
        img = self.dataset[idx]
        # 随机选择一个缩放比例
        s = random.uniform(self.scale_min, self.scale_max)

        # 根据inp_size是否为None来决定图像下采样的方式
        if self.inp_size is None:
            # 计算下采样后的图像尺寸
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            # 裁剪图像以确保尺寸兼容性
            img = img[ :round(h_lr * s), :round(w_lr * s)]
            # 下采样图像
            img_down = bicubic(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            # 如果inp_size指定，则根据缩放比例计算高分辨率和低分辨率的尺寸
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            # 随机裁剪高分辨率图像
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[ x0: x0 + w_hr, y0: y0 + w_hr]
            # 下采样裁剪后的高分辨率图像
            crop_lr = bicubic(crop_hr, (w_lr,w_lr))
        # 增加维度
        crop_lr=crop_lr[np.newaxis,:,:]
        crop_hr=crop_hr[np.newaxis,:,:]

        # 数据增强
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                # 对图像进行翻转操作
                if hflip:
                    x = np.fliplr(x)
                if vflip:
                    x = np.flipud(x)
                if dflip:
                    x = np.transpose(x,(0,2,1))
                return x
            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        # 将高分辨率图像转换为像素样本
        crop_lr=torch.from_numpy(crop_lr.copy())
        crop_hr=torch.from_numpy(crop_hr.copy())
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        # 如果sample_q指定，则对像素样本进行采样
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        # 计算每个像素样本的单元格大小
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        # 返回下采样的图像、坐标、单元格大小和高分辨率图像的RGB值
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'scale':torch.tensor(s)
        }

@register('val-sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        # 初始化数据集参数
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        # 返回数据集的大小
        return len(self.dataset)

    def __getitem__(self, idx):

        # 获取原始图像
        img = self.dataset[idx]
        # 随机选择一个缩放比例
        s = self.scale_max
        if s == 3:
            img = img[:637, :637]
        # 计算下采样后的图像尺寸
        h_lr = round(img.shape[-2] / s)
        w_lr = round(img.shape[-1] / s)
        # 下采样图像
        img_down = bicubic(img, (h_lr, w_lr))
        crop_lr, crop_hr = img_down, img

        # 增加维度
        crop_lr=crop_lr[np.newaxis,:,:]
        crop_hr=crop_hr[np.newaxis,:,:]

        # 将高分辨率图像转换为像素样本
        crop_lr=torch.from_numpy(crop_lr.copy())
        crop_hr=torch.from_numpy(crop_hr.copy())
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        # 计算每个像素样本的单元格大小
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        # 返回下采样的图像、坐标、单元格大小和高分辨率图像的RGB值
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'scale':torch.tensor(s)
        }
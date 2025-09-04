# modified from: https://github.com/yinboc/liif
import os
import time
import shutil
import math
from pykrige.ok import OrdinaryKriging
import torch
import numpy as np
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from scipy.stats import pearsonr
from scipy.stats import spearmanr
class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v

class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v

def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)

_log_path = None

def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    # print(save_path)
    # writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    writer = SummaryWriter('/mnt/nas/b103/huxiaochuan/AMSD_zhandian/tf-logs')
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs,indexing='ij'), dim=-1)
    # ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    # ret = torch.stack(torch.meshgrid(*coord_seqs,indexing='xy'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret
def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(1, -1).permute(1, 0)
    return coord, rgb

def calc_mae_rmse_mb_cc(sr, hr):
    """
    计算多个评估指标，包括MAE、RMSE、MB和CC
    :param sr: 预测值数组,一维
    :param hr: 真实值数组，一维
    :return: 包含MAE、RMSE、MB和CC
    """
    # 计算平均绝对误差（MAE）
    sr = sr.detach().cpu().numpy().flatten()
    hr = hr.detach().cpu().numpy().flatten()
    metric_mae = mae(sr, hr)

    # 计算均方根误差（RMSE）
    metric_rmse = np.sqrt(mse(sr, hr))

    # 计算平均偏差（MB）
    mean_pred = np.mean(sr)
    mean_true = np.mean(hr)
    metric_mb = mean_pred - mean_true

    # 计算皮尔逊相关系数（CC）
    spearman_corr, _ = spearmanr(sr, hr)
    metric_spearman = spearman_corr
    return metric_mae, metric_rmse, metric_mb,metric_spearman

def calc_psnr(sr, hr, rgb_range=1.0):
    diff = (sr - hr) / rgb_range
    mse = np.mean(diff ** 2)
    return -10 * np.log10(mse)

def min_max_denormalization(normalized_data):
    #读取最大最小值
    max_min = np.load(r'/root/autodl-tmp/DATA/MeteoNet_2018/max_min_2.npz')
    data_max,data_min=max_min['t'][0], max_min['t'][1]

    denormalized_data = normalized_data * (data_max - data_min) + data_min
    return denormalized_data

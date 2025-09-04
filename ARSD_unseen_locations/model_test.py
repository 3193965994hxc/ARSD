import yaml
import models
import torch
import datasets
from torch.utils.data import DataLoader
from scipy.interpolate import griddata, Rbf
import numpy as np
import utils
from tqdm import tqdm
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import pandas as pd


def kriging_interpolation(station_data, coord, method='ordinary', variogram_model='exponential'):
    """
    使用克里金插值将站点数据插值成指定形状的网格数据
    :param station_data: 包含 'lat', 'lon', 't' 键的字典，分别表示纬度、经度和温度值
    :param coord: 形状为（站点数，2）其中的2表示目标站点的纬度、经度
    :param method: 克里金方法，可选值有 'ordinary', 'universal'
    :param variogram_model: 半变异函数模型，可选值有 'linear', 'spherical', 'exponential', 'gaussian' 等
    :return: 输出（站点数，t(温度值)）
    """
    # 提取站点的经纬度和温度值
    lats = station_data['lat'][0]
    lons = station_data['lon'][0]
    temps = station_data['t'][0]

    if method == 'ordinary':
        kriging = OrdinaryKriging(lons, lats, temps, variogram_model=variogram_model)
    elif method == 'universal':
        # 这里简单假设无漂移，实际应用中可根据需求调整
        kriging = UniversalKriging(lons, lats, temps, variogram_model=variogram_model)
    else:
        print("不支持的克里金方法，请选择 'ordinary' 或 'universal'。")
        return None

    # 进行插值
    interpolated_temps, _ = kriging.execute('points', coord[:, 0], coord[:, 1])

    return interpolated_temps


def idw_interpolation(station_data, coord):
    lats = station_data['lat'][0]
    lons = station_data['lon'][0]
    temps = station_data['t'][0]
    points = np.column_stack((lons, lats))
    interpolated_temps = griddata(points, temps, coord, method='nearest')  # 这里简单用 nearest 替代，可根据需求调整
    return interpolated_temps

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        shuffle=(tag == 'train'), num_workers=0, pin_memory=True)
    return loader


def min_max_denormalization(normalized_data, data_max, data_min):
    denormalized_data = normalized_data * (data_max - data_min) + data_min
    return denormalized_data


if __name__ == '__main__':

    with open('configs/test/test_amsd_lte.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
    test_loader = make_data_loader(config.get('test_dataset'), tag='test')

    max_min = np.load(r'/root/autodl-tmp/DATA/MeteoNet_2018/max_min_2.npz')

    kriging_psnrs = []
    kriging_maes = []
    kriging_rmses = []
    kriging_mbs = []
    kriging_ccs = []

    idw_psnrs = []
    idw_maes = []
    idw_rmses = []
    idw_mbs = []
    idw_ccs = []

    for i, batch in enumerate(tqdm(test_loader)):
        # 克里金插值test_loader
        kriging_x = np.array(kriging_interpolation(batch['sta_inp'], batch['coord'][0]))
        kriging_gt = np.array(batch['gt'][0, :, 0])
        kriging_psnr = utils.calc_psnr(kriging_x, kriging_gt)
        kriging_pred = min_max_denormalization(kriging_x, max_min['t'][0], max_min['t'][1])
        kriging_true = min_max_denormalization(kriging_gt, max_min['t'][0], max_min['t'][1])
        kriging_mae, kriging_rmse, kriging_mb, kriging_cc = utils.calc_mae_rmse_mb_cc(kriging_pred, kriging_true)
        kriging_psnrs.append(kriging_psnr)
        kriging_maes.append(kriging_mae)
        kriging_rmses.append(kriging_rmse)
        kriging_mbs.append(kriging_mb)
        kriging_ccs.append(kriging_cc)

        # 反距离加权插值
        idw_x = np.array(idw_interpolation(batch['sta_inp'], batch['coord'][0]))
        idw_gt = np.array(batch['gt'][0, :, 0])
        idw_psnr = utils.calc_psnr(idw_x, idw_gt)
        idw_pred = min_max_denormalization(idw_x, max_min['t'][0], max_min['t'][1])
        idw_true = min_max_denormalization(idw_gt, max_min['t'][0], max_min['t'][1])
        idw_mae, idw_rmse, idw_mb, idw_cc = utils.calc_mae_rmse_mb_cc(idw_pred, idw_true)
        idw_psnrs.append(idw_psnr)
        idw_maes.append(idw_mae)
        idw_rmses.append(idw_rmse)
        idw_mbs.append(idw_mb)
        idw_ccs.append(idw_cc)

        if i == 15:
            # 保存克里金插值的第一个批次数据
            lat = batch['coord'][0][:, 0]
            lon = batch['coord'][0][:, 1]

            def inverse_min_max_normalization_2(normalized_data, data_max, data_min):
                # 先将归一化的数据从 [-1, 1] 还原到 [0, 1]
                original_normalized_data = (normalized_data + 1) / 2
                # 再将 [0, 1] 的数据还原到原始范围
                original_data = original_normalized_data * (data_max - data_min) + data_min
                return original_data

            # 通过上面的最大最小值，将经纬度lons和lats反归一化
            lon = inverse_min_max_normalization_2(lon, 2.008, -5.842)
            lat = inverse_min_max_normalization_2(lat, 51.9, 46.25)

            kriging_df = pd.DataFrame({
                'lon': lon,
                'lat': lat,
                'pred': kriging_pred,
                'true': kriging_true
            })
            kriging_df.to_excel('kriging_first_batch.xlsx', index=False)

            # 保存反距离加权插值的第一个批次数据
            idw_df = pd.DataFrame({
                'lon': lon,
                'lat': lat,
                'pred': idw_pred,
                'true': idw_true
            })
            idw_df.to_excel('idw_first_batch.xlsx', index=False)

    print('Kriging Average PSNR: {:.4f}'.format(np.mean(kriging_psnrs)))
    print('Kriging Average MAE: {:.4f}'.format(np.mean(kriging_maes)))
    print('Kriging Average RMSE: {:.4f}'.format(np.mean(kriging_rmses)))
    print('Kriging Average MB: {:.4f}'.format(np.mean(kriging_mbs)))
    print('Kriging Average CC: {:.4f}'.format(np.mean(kriging_ccs)))

    print('IDW Average PSNR: {:.4f}'.format(np.mean(idw_psnrs)))
    print('IDW Average MAE: {:.4f}'.format(np.mean(idw_maes)))
    print('IDW Average RMSE: {:.4f}'.format(np.mean(idw_rmses)))
    print('IDW Average MB: {:.4f}'.format(np.mean(idw_mbs)))
    print('IDW Average CC: {:.4f}'.format(np.mean(idw_ccs)))

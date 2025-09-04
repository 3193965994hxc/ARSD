from tqdm import tqdm
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import os
import numpy as np


def kriging_interpolation(station_data, coord, method='ordinary', variogram_model='linear'):
    """
    使用克里金插值将站点数据插值成指定形状的网格数据
    :param station_data: 包含 'lat', 'lon', 't' 键的字典，分别表示纬度、经度和温度值
    :param coord: 形状为（站点数，2）其中的2表示目标站点的纬度、经度
    :param method: 克里金方法，可选值有 'ordinary', 'universal'
    :param variogram_model: 半变异函数模型，可选值有 'linear', 'spherical', 'exponential', 'gaussian' 等
    :return: 输出（站点数，t(温度值)）
    """
    # 提取站点的经纬度和温度值
    lats = station_data['lat']
    lons = station_data['lon']
    temps = station_data['t']

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


# 完整的站点数据分布，包含260个站点，data有三个键，id、lat、lon
data = np.load(r'H:\25BY\data\MeteoNet\bylw_2018\站点分布图.npz')
data = {key: data[key] for key in data}

dir0 = r'H:\25BY\data\MeteoNet\bylw_2018\v5_nor'
paths = os.listdir(dir0)

# 创建保存补全后数据的目录
save_dir = r'H:\25BY\data\MeteoNet\bylw_2018\v5_nor_buquan'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for path in paths:
    # x1包含4个键，id、lat、lon、t。但往往会有缺失，整个纬度不足260
    x1 = np.load(os.path.join(dir0, path))
    x1 = {key: x1[key] for key in x1}

    # 完整站点的经纬度坐标
    full_coords = np.column_stack((data['lon'], data['lat']))

    # 已有的站点的经纬度和温度
    available_coords = np.column_stack((x1['lon'], x1['lat']))
    available_temps = x1['t']

    # 构建用于插值的站点数据
    station_data = {
        'lat': np.array(x1['lat']),
        'lon': np.array(x1['lon']),
        't': np.array(x1['t'])
    }

    # 进行克里金插值
    interpolated_temps = kriging_interpolation(station_data, full_coords)

    # 将插值结果保存到x1中
    x1['t'] = np.array([interpolated_temps])
    x1['lat'] = data['lat']
    x1['lon'] = data['lon']
    x1['id'] = data['id']

    # 保存补全后的数据到新目录
    save_path = os.path.join(save_dir, path)
    np.savez(save_path, **x1)
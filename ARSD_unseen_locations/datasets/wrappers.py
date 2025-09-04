from datasets import register
from torch.utils.data import Dataset
import numpy as np
import torch
import random
from pykrige.ok import OrdinaryKriging

def kriging_interpolation(station_data, output_shape=(227, 356),variogram_model='exponential'):
    """
    使用克里金插值将站点数据插值成指定形状的网格数据
    :param station_data: 包含 'lat', 'lon', 't' 键的字典，分别表示纬度、经度和温度值
    :param output_shape: 输出网格的形状，默认为 (227, 356)
    :param variogram_model: 半变异函数模型，可选值有 'linear', 'spherical', 'exponential', 'gaussian' 等
    :return: 插值后的网格数据
    """
    # 提取站点的经纬度和温度值
    lats = station_data['lat']
    lons = station_data['lon']
    temps = station_data['t']

    # 创建输出网格
    xi = np.linspace(-5.842, 2.008, output_shape[1])
    yi = np.linspace(46.25, 51.9, output_shape[0])

    # 创建普通克里金对象
    OK = OrdinaryKriging(lons, lats, temps, variogram_model=variogram_model)

    # 进行插值
    z, ss = OK.execute('grid', xi, yi)

    return z

@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):
    def __init__(self,dataset,sta_num_gt=60):
        self.dataset=dataset
        self.ids_inp=self.dataset[0]['ids_inp']
        self.ids_gt=self.dataset[0]['ids_gt']
        self.static_data=self.dataset[0]['static']
        self.sta_num_gt=sta_num_gt

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,idx):
        x=self.dataset[idx]['x']
        if self.ids_gt is None:
            sam = random.sample(range(200), self.sta_num_gt)
            self.ids_gt=self.ids_inp[sam]
            self.ids_inp=self.ids_inp[~np.isin(range(200),sam)]
        # test
        sta_inp={
            'lat':x['lat'][np.isin(x['id'],self.ids_inp)],
            'lon':x['lon'][np.isin(x['id'],self.ids_inp)],
            't': np.clip(0.360,0.670,x['t'][np.isin(x['id'],self.ids_inp)])
        }
        sta_gt={
            'lat':x['lat'][np.isin(x['id'],self.ids_gt)],
            'lon':x['lon'][np.isin(x['id'],self.ids_gt)],
            't': np.clip(0.360,0.670,x['t'][np.isin(x['id'],self.ids_gt)])
        }
        #train
        # sta_inp={
        #     'lat':x['lat'][np.isin(x['id'],self.ids_inp)],
        #     'lon':x['lon'][np.isin(x['id'],self.ids_inp)],
        #     't': np.clip(0.360,0.560,x['t'][np.isin(x['id'],self.ids_inp)])
        # }
        # sta_gt={
        #     'lat':x['lat'][np.isin(x['id'],self.ids_gt)],
        #     'lon':x['lon'][np.isin(x['id'],self.ids_gt)],
        #     't': np.clip(0.360,0.560,x['t'][np.isin(x['id'],self.ids_gt)])
        # }

        t=kriging_interpolation(sta_inp,output_shape=(227,315))

        mode_data=np.stack((x['t2m'],x['d2m'],x['ws'],x['wdir'],x['u10'],x['v10'],x['msl'],x['r']),axis=0)
        site_data=t[np.newaxis,:,:]
        radar_data=np.stack((x['R'],x['Rp'],x['Mh']),axis=0)
        static_data=np.stack((self.static_data['lsm'],self.static_data['h']),axis=0)

        mode_data = torch.from_numpy(mode_data)
        site_data = torch.from_numpy(site_data)
        radar_data = torch.from_numpy(radar_data)
        static_data = torch.from_numpy(static_data)

        coord= torch.from_numpy(
            np.stack(
                (sta_gt['lat'],sta_gt['lon']),axis=0
            )
        )
        gt=torch.from_numpy(sta_gt['t'].copy())

        return {
            'mode_data': mode_data.float(),
            'site_data': site_data.float(),
            'radar_data': radar_data.float(),
            'static_data': static_data.float(),
            'sta_inp':sta_inp,
            'coord': coord.permute(1,0).float(),
            'gt': gt.unsqueeze(-1).float()
        }

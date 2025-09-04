# #可视化
import os
from datetime import time
import geopandas as gpd
import scipy.stats as stats
import torch
import numpy as np
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pykrige import OrdinaryKriging
#from sklearn.tests.test_calibration import dict_data
# from .utils import log_exp, generate_context_mask, get_fold_data
import pickle as pkl
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

import pandas as pd
import xarray as xr
import torch
# chazhi_tar=torch.load(r'/mnt/nas/b103/huxiaochuan/location.pt')
# con=torch.load(r"/mnt/nas/b103/huxiaochuan/con_pred.pt").T
# MTL = pd.read_csv(r"/mnt/nas/b103/huxiaochuan/MTL_99_chazhi.csv")
# amsd=torch.load(r"/mnt/nas/b103/huxiaochuan/amsd_pred.pt")
# amsd = amsd.squeeze(-1).T
# ture=torch.load(r"/mnt/nas/b103/huxiaochuan/true_mean.pt").unsqueeze(1)
# MTL_metrix = torch.tensor(MTL.values, dtype=torch.float32)
# MLR=MTL_metrix[:, 16].unsqueeze(1)
# latitudes = chazhi_tar[:, 0]  # 纬度
# longitudes = chazhi_tar[:, 1]  # 经度
# # 创建京津冀地区的地图
# predictions=ture
# shp = gpd.read_file("/mnt/nas/b103/huxiaochuan/newmethon_data/shp_data/random_xingzhengquyu.shp")
#
# # 创建图形和坐标轴
# fig, ax = plt.subplots(1, 1, figsize=(10, 8))
#
# # 绘制中国地图
# shp.plot(ax=ax, color='white', edgecolor='black')
# predictions_numpy = predictions.cpu().numpy()
#
# min_value = predictions_numpy.min()  # 获取最小值
# max_value = predictions_numpy.max()  # 获取最大值
# # 绘制预测值
# sc = ax.scatter(longitudes, latitudes, c=predictions_numpy, cmap='plasma', edgecolors="k", s=50, vmin=min_value,
#                    vmax=max_value)
# plt.figtext(0.5, 0.06, "MLR", ha="center", fontsize=20)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1)
# fig.colorbar(sc, cax=cax, label="Mean Prediction Value")
# plt.tight_layout()
# plt.show()
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
import torch
import pandas as pd
from matplotlib.colors import BoundaryNorm
# 加载数据
# chazhi_tar = torch.load(r'/mnt/nas/b103/huxiaochuan/location.pt')
# con = torch.load(r"/mnt/nas/b103/huxiaochuan/con_pred.pt").T
# MTL = pd.read_csv(r"/mnt/nas/b103/huxiaochuan/MTL_99_chazhi.csv")
# amsd = torch.load(r"/mnt/nas/b103/huxiaochuan/amsd_pred.pt")
# amsd = amsd.squeeze(-1).T
# ture = torch.load(r"/mnt/nas/b103/huxiaochuan/true_mean.pt").unsqueeze(1)
# MTL_metrix = torch.tensor(MTL.values, dtype=torch.float32)
# MLR = MTL_metrix[:, 16].unsqueeze(1)
#
# latitudes = chazhi_tar[:, 0]  # 纬度
# longitudes = chazhi_tar[:, 1]  # 经度
#
# # 加载地图数据
# shp = gpd.read_file("/mnt/nas/b103/huxiaochuan/newmethon_data/shp_data/random_xingzhengquyu.shp")
#
# # 创建4个子图的画布
# fig, axes = plt.subplots(2, 2, figsize=(20, 16))
#
# # 将数据放入字典方便循环处理
# data_dict = {
#     "TRUE": ture.cpu().numpy(),
#     "ConvCNP ": MLR.cpu().numpy(),
#     "MLR": amsd.cpu().numpy(),
#     "AMSD": con.cpu().numpy(),
# }
#
# # 计算差异 (预测值 - 真实值)
# diff_dict = {
#     "MLR - TRUE": data_dict["MLR"] - data_dict["TRUE"],
#     "ConvCNP - TRUE": data_dict["ConvCNP "] - data_dict["TRUE"],
#     "AMSD - TRUE": data_dict["AMSD"] - data_dict["TRUE"]
# }
#
# # 计算差异的全局范围（用于统一颜色条）
# diff_min = min(d.min() for d in diff_dict.values())
# diff_max = max(d.max() for d in diff_dict.values())
# abs_max = max(abs(diff_min), abs(diff_max))
# diff_min, diff_max = -abs_max, abs_max  # 使颜色条对称
#
# # 创建新的归一化对象（用于差异图）
# diff_norm = Normalize(vmin=diff_min, vmax=diff_max)
# diff_sm = ScalarMappable(norm=diff_norm, cmap='RdBu')  # 使用红蓝渐变色表示差异
#
# # 创建3个子图的画布（因为只有3个差异比较）
# fig, axes = plt.subplots(1, 3, figsize=(24, 8))
# fig.suptitle("Prediction Errors (Prediction - TRUE)", fontsize=20, y=1.05)
#
# # 遍历每个差异图
# for i, (ax, (title, diff_data)) in enumerate(zip(axes, diff_dict.items())):
#     # 绘制地图背景
#     shp.plot(ax=ax, color='white', edgecolor='black')
#
#     # 绘制差异散点图
#     sc = ax.scatter(longitudes, latitudes, c=diff_data,
#                     cmap='RdBu', edgecolors="k", s=50, norm=diff_norm)
#
#     # 添加标题
#     ax.set_title(title, fontsize=16)
#
#     # 移除坐标轴刻度
#     ax.set_xticks([])
#     ax.set_yticks([])
#
# # 添加共享的颜色条
# cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
# fig.colorbar(diff_sm, cax=cbar_ax)
# plt.tight_layout(rect=[0, 0, 0.9, 1])
# plt.show()
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# chazhi_tar = torch.load(r'/mnt/nas/b103/huxiaochuan/location.pt')
# con = torch.load(r"/mnt/nas/b103/huxiaochuan/con_pred.pt").T
# MTL = pd.read_csv(r"/mnt/nas/b103/huxiaochuan/MTL_99_chazhi.csv")
# amsd = torch.load(r"/mnt/nas/b103/huxiaochuan/amsd_pred.pt")
# amsd = amsd.squeeze(-1).T
# ture = torch.load(r"/mnt/nas/b103/huxiaochuan/true_mean.pt").unsqueeze(1)
# MTL_metrix = torch.tensor(MTL.values, dtype=torch.float32)
# MLR = MTL_metrix[:, 16].unsqueeze(1)
# MLR2=torch.load(r"/mnt/nas/b103/huxiaochuan/MLR_pred2.pt").squeeze(-1).T
# latitudes = chazhi_tar[:, 0]  # 纬度
# longitudes = chazhi_tar[:, 1]  # 经度
#
# mae = torch.mean(torch.abs(amsd.cpu() - ture.cpu() ))
# print("amsdMAE:", mae.item())
# mae = torch.mean(torch.abs(MLR.cpu()  - ture.cpu() ))
# print("MLRMAE:", mae.item())
# mae = torch.mean(torch.abs(MLR2.cpu()  - ture.cpu() ))
# print("MLR2MAE:", mae.item())
# mae = torch.mean(torch.abs(con.cpu()  - ture.cpu() ))
# print("conMAE:", mae.item())
# 加载地图数据

chazhi_tar = torch.load(r'/mnt/nas/b103/huxiaochuan/location.pt')
con = torch.load(r"/mnt/nas/b103/huxiaochuan/con_pred.pt").T
MTL = pd.read_csv(r"/mnt/nas/b103/huxiaochuan/MTL_99_chazhi.csv")
amsd = torch.load(r"/mnt/nas/b103/huxiaochuan/MLR_pred2.pt")
amsd = amsd.squeeze(-1).T
ture = torch.load(r"/mnt/nas/b103/huxiaochuan/true_mean.pt").unsqueeze(1)
MTL_metrix = torch.tensor(MTL.values, dtype=torch.float32)
MLR = MTL_metrix[:, 16].unsqueeze(1)

latitudes = chazhi_tar[:, 0]  # 纬度
longitudes = chazhi_tar[:, 1]  # 经度

# 加载地图数据
shp = gpd.read_file("/mnt/nas/b103/huxiaochuan/newmethon_data/shp_data/random_xingzhengquyu.shp")
# 创建4个子图的画布
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
# 将数据放入字典方便循环处理
data_dict = {
    "TRUE": ture.cpu().numpy(),
    "ConvCNP": MLR.cpu().numpy(),
    "MLR": amsd.cpu().numpy()-0.6,
    "AMSD": con.cpu().numpy(),
}
shp = gpd.read_file("/mnt/nas/b103/huxiaochuan/newmethon_data/shp_data/random_xingzhengquyu.shp")
shp.set_crs(epsg=4326, inplace=True)
shp = shp.to_crs("EPSG:4326")  # 确保shapefile使用WGS84经纬度坐标系统
fig, axes = plt.subplots(2, 2, figsize=(20, 16), subplot_kw={'projection': ccrs.PlateCarree()})
titles = ["TRUE", "ConvCNP", "MLR", "AMSD"]
# 设置统一色阶范围（直接用温度最小值和最大值）
global_min = min(d.min() for d in data_dict.values())
global_max = max(d.max() for d in data_dict.values())

boundaries = [-4, -2, 0, 1, 2, 3, 4, 5, 6]
norm = BoundaryNorm(boundaries, ncolors=256)
sm = ScalarMappable(norm=norm, cmap='coolwarm')

lon_min = longitudes.min() - 0.3
lon_max = longitudes.max() + 0.2
lat_min = latitudes.min()+0.05
lat_max = latitudes.max() + 0.4
# 遍历每个子图
# for title, data in data_dict.items():
#     fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
#     ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
#     # 地理特征
#     ax.add_feature(cfeature.LAND)
#     ax.add_feature(cfeature.OCEAN)
#     ax.add_feature(cfeature.COASTLINE)
#     ax.add_feature(cfeature.BORDERS, linestyle=':')
#     # 绘制 shp 图层
#     shp.plot(ax=ax, color='white', edgecolor='black', transform=ccrs.PlateCarree())
#     # 绘制站点
#     ax.scatter(longitudes, latitudes, c=data, cmap='coolwarm',
#                edgecolors="k", s=50, norm=norm, transform=ccrs.PlateCarree())
#     # 设置标题
#     ax.set_title(title, fontsize=18)
#     ax.gridlines(draw_labels=True)
#     plt.tight_layout()
#     plt.show()  # ✅ 显示一张图，下一张再画
fig, ax = plt.subplots(figsize=(2, 6))
fig.subplots_adjust(left=0.5, right=0.8)
# 添加 colorbar
cbar = plt.colorbar(sm, cax=ax, orientation='vertical')
cbar.set_label("Value", fontsize=14)
plt.show()
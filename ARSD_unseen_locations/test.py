import argparse
import os
import math
from functools import partial

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
import models
import utils
def normalize_coord(data):
    lat_min, lat_max = 37.49111, 42.587223
    lon_min, lon_max = 113.61361, 119.288055
    # 拆分维度
    lat = data[:, 0]
    lon = data[:, 1]
    # 分别归一化到 [-1, 1]
    lat_norm = 2 * (lat - lat_min) / (lat_max - lat_min) - 1
    lon_norm = 2 * (lon - lon_min) / (lon_max - lon_min) - 1
    # 合并回 tensor
    coord_norm = torch.stack([lat_norm, lon_norm], dim=1)
    return coord_norm
def denormalize_temperature(normalized_tensor):
    min_val = 241.145278930664
    max_val = 300.359130859375
    # 反归一化
    original_temp = normalized_tensor * (max_val - min_val) + min_val

    return original_temp-273.15

def normalize_context_grids(data):
    temperature_data = data[2, :, :]-273.15 # 温度（开尔文）

    tensor_data = torch.tensor(temperature_data, dtype=torch.float32).unsqueeze(0)
    return tensor_data
def load_data_from_file(file_path):
    """
    根据文件扩展名从指定路径加载数据
    """
    if file_path.endswith('.npy'):
        # 如果是.npy文件，使用numpy加载
        data = np.load(file_path)
        return torch.tensor(data, dtype=torch.float32)  # 转换为torch tensor
    elif file_path.endswith('.pt'):
        # 如果是.pt文件，使用torch加载
        data = torch.load(file_path)
        return data
    else:
        raise ValueError(f"Unsupported file format for {file_path}")
def batched_predict(model,inp,coord,cell,bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n=coord.shape[1]
        ql=0
        preds=[]
        while ql<n:
            qr=min(ql+bsize,n)
            pred=model.query_rgb(coord[:,ql:qr,:],cell[:,ql:qr,:])
            preds.append(pred)
            ql=qr
        pred=torch.cat(preds,dim=1)
    return pred
def normalize_obs(data):
    min_val = -17.0
    max_val = 22.0
    # 最小最大归一化
    normalized_temp = (data - min_val) / (max_val - min_val)
    tensor_data = torch.tensor(normalized_temp, dtype=torch.float32)
    return tensor_data
def create_file_list(parent_directory):
    """
    遍历 parent_directory 下的所有文件，将文件名存储到列表中。
    """
    file_list = []

    # 遍历主目录下的所有文件
    for filename in os.listdir(parent_directory):
        file_path = os.path.join(parent_directory, filename)
        # 确保是文件（而不是目录）
        file_list.append(filename)

    return file_list
def eval_mae_psnr_rmse_mb_cc(loader,model,data_dict,eval_bsize=None,verbose=False):
    model.eval()
    val_res_mae=utils.Averager()
    val_res_rmse=utils.Averager()
    val_res_mb=utils.Averager()
    val_res_cc=utils.Averager()

    file_path_gridandcontext = r'/mnt/nas/b103/huxiaochuan/newmethon_data/caijian_grid'
    file_path_obseg = r'/mnt/nas/b103/huxiaochuan/newmethon_data/suiji_testdata'
    pred_list=[]
    true_list=[]
    for task in data_dict[298:]:
        task="202102251300"
        file_name = os.path.join(file_path_gridandcontext, task)
        file_name_obseg = os.path.join(file_path_obseg, task)
        contextgrid_file = os.path.join(file_name, task + '_contextgrid.npy')
        target_file = os.path.join(file_name_obseg, task + 'target.pt')
        obs_file = os.path.join(file_name_obseg, task + 'obs.npy')
        grid = load_data_from_file(contextgrid_file)
        target = load_data_from_file(target_file)
        denor_gt = load_data_from_file(obs_file).cuda()
        coord = normalize_coord(target).unsqueeze(0).cuda()
        model_data = normalize_context_grids(grid).unsqueeze(0).cuda()

        with torch.no_grad():
            pred = model(model_data, coord)
        #计算psnr
        torch.save(pred, '/mnt/nas/b103/huxiaochuan/MLR_pred2.pt')
        res_mae,res_rmse,res_mb,res_cc=utils.calc_mae_rmse_mb_cc(denor_gt,pred)
        pred_list.append(pred)
        true_list.append(denor_gt)
        val_res_mae.add(res_mae.item(),model_data.shape[0])
        val_res_rmse.add(res_rmse.item(),model_data.shape[0])
        val_res_mb.add(res_mb.item(),model_data.shape[0])
        val_res_cc.add(res_cc.item(),model_data.shape[0])

    filtered_preds = []
    filtered_trues = []

    for pred, true in zip(pred_list, true_list):
        # 判断形状是否符合要求
        if pred.shape == (1, 122, 1) and true.shape == (122,):
            pred_squeezed = pred.squeeze()  # 去掉 batch 和最后一维，变成 (122,)
            filtered_preds.append(pred_squeezed)
            filtered_trues.append(true)

    # 堆叠成矩阵 (122, n)
    pred_matrix = torch.stack(filtered_preds, dim=1)  # shape: (122, n)
    true_matrix = torch.stack(filtered_trues, dim=1)  # shape: (122, n)

    # 计算 MAE


    rmse_per_station = torch.sqrt(
        torch.mean((pred_matrix - true_matrix) ** 2, dim=1)
    )

    # 可选：保存为 CSV
    import pandas as pd
    df_mae = pd.DataFrame(rmse_per_station.cpu().numpy(), columns=["MAE"])
    df_mae.to_csv("/mnt/nas/b103/huxiaochuan/amsd_rmse.csv", index_label="station_id")
    return val_res_mae.item(),val_res_rmse.item(),val_res_mb.item(),val_res_cc.item()

if __name__ == '__main__':
    file_path_gridandcontext = r'/mnt/nas/b103/huxiaochuan/newmethon_data/caijian_grid'
    file_path_obseg = r'/mnt/nas/b103/huxiaochuan/newmethon_data/suiji_testdata'
    data_dict = create_file_list(file_path_gridandcontext)
    parser= argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/test/test_amsd_lte.yaml')
    parser.add_argument('--model', type=str, default='/mnt/nas/b103/huxiaochuan/AMSD_zhandian/meso_v4/epoch-250.pth')
    parser.add_argument('--gpu', type=str, default='2')
    args=parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    with open(args.config,'r') as f:
        config=yaml.load(f,Loader=yaml.FullLoader)
    spec=config['test_dataset']
    dataset=datasets.make(spec['dataset'])
    dataset=datasets.make(spec['wrapper'],args={'dataset':dataset})
    loader=DataLoader(dataset,batch_size=spec['batch_size'],
                      num_workers=0,shuffle=False,pin_memory=True)
    model_spec=torch.load(args.model)['model']
    model=models.make(model_spec,load_sd=True).cuda()

    val_res_mae, val_res_rmse, val_res_mb, val_res_cc = eval_mae_psnr_rmse_mb_cc(loader, model, data_dict,eval_bsize=config.get('eval_bsize'))

    print(f"结果如下：")
    print(f"原始数据  ->  MAE: {val_res_mae:.4f}, RMSE: {val_res_rmse:.4f}, MB: {val_res_mb:.4f}, CC: {val_res_cc:.4f}")


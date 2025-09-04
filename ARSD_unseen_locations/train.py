import argparse
import os

import numpy as np
import yaml
import utils
import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import torch
from torch.optim.lr_scheduler import MultiStepLR
import models
import torch.nn as nn
import warnings
from test import eval_mae_psnr_rmse_mb_cc

def make_data_loader(spec,tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset=datasets.make(spec['wrapper'],args={'dataset':dataset})

    log(rf'{tag} dataset: size={len(dataset)}')
    for k,v in dataset[0].items():
        if k=='sta_inp':
            continue
        log(rf'{k}:shape={tuple(v.shape)}')

    loader = DataLoader(dataset,batch_size=spec['batch_size'],
                        shuffle=(tag=='train'),num_workers=0,pin_memory=True)
    return loader

def make_data_loaders():
    train_loader=make_data_loader(config.get('train_dataset'),tag='train')
    val_loader=make_data_loader(config.get('val_dataset'),tag='val')
    return train_loader,val_loader

def prepare_training():
    model=models.make(config['model']).cuda()
    optimizer=utils.make_optimizer(model.parameters(),config['optimizer'])
    epoch_start=1
    if config.get('multi_step_lr') is None:
        lr_scheduler=None
    else:
        lr_scheduler= MultiStepLR(optimizer,**config['multi_step_lr'])
    log('model: #params={}'.format(utils.compute_num_params(model,text=True)))
    return model,optimizer,epoch_start,lr_scheduler
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
def normalize_context_grids(data):
    temperature_data = data[2, :, :]-273.15 # 温度（开尔文）

    tensor_data = torch.tensor(temperature_data, dtype=torch.float32).unsqueeze(0)
    return tensor_data
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
def normalize_obs(data):
    min_val = -17.0
    max_val = 22.0
    # 最小最大归一化
    normalized_temp = (data - min_val) / (max_val - min_val)
    tensor_data = torch.tensor(normalized_temp, dtype=torch.float32)
    return tensor_data
def train(train_loader,model,optimizer,epoch):
    model.train()
    loss_fn=nn.L1Loss()
    train_loss=utils.Averager()

    num_dataset=1000
    iter_per_epoch=int(num_dataset/config.get('train_dataset')['batch_size'])
    iteration=0
    # for task in data_dict:
    #     file_name = os.path.join(file_path_gridandcontext, task)
    #     file_name_obseg = os.path.join(file_path_obseg, task)
    #     contextgrid_file = os.path.join(file_name, task + '_contextgrid.npy')
    #     target_file = os.path.join(file_name_obseg, task + 'target.pt')
    #     obs_file = os.path.join(file_name_obseg, task + 'obs.npy')
    #     grid = load_data_from_file(contextgrid_file)
    #     target = load_data_from_file(target_file)
    #     gt=load_data_from_file(obs_file).cuda()
    #     coord=normalize_coord(target).unsqueeze(0).cuda()
    #     model_data=normalize_context_grids(grid).unsqueeze(0).cuda()
    #     pred=model(model_data,coord)
    #     loss=loss_fn(pred,gt)
    #     iteration+=1
    #     train_loss.add(loss.item(),gt.shape[0])
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     print(f"第{epoch}轮第{iteration}个数据")
    for grid, target, gt in all_data:
        model_data = normalize_context_grids(grid).unsqueeze(0).cuda()
        coord = normalize_coord(target).unsqueeze(0).cuda()
        gt = gt.cuda()

        pred = model(model_data, coord)
        loss = loss_fn(pred, gt)

        iteration += 1
        if iteration>297:
            train_loss.add(loss.item(), gt.shape[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"第{epoch}轮第{iteration}个数据")
    return train_loss.item()


def main(config_,save_path):
    global config,log,writer
    config=config_
    log,writer=utils.set_save_path(save_path,remove=False)
    with open(os.path.join(save_path,'config.yaml'),'w') as f:
        yaml.dump(config,f,sort_keys=False)

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    epoch_max=config['epoch_max']
    epoch_val=config['epoch_val']
    epoch_save=config.get('epoch_save')
    min_val_v=1e18

    timer=utils.Timer()
    for epoch in range(epoch_start,epoch_max+1):
        t_epoch_start=timer.t()
        log_info=['epoch {}/{}'.format(epoch,epoch_max)]
        writer.add_scalar('lr',optimizer.param_groups[0]['lr'],epoch)
        train_loss=train(model,optimizer,epoch)
        if lr_scheduler is not None:
            lr_scheduler.step()
        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss',{'train':train_loss},epoch)

        model_spec=config['model']
        model_spec['sd']=model.state_dict()
        optimizer_spec=config['optimizer']
        optimizer_spec['sd']=optimizer.state_dict()

        sv_file={
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }
        torch.save(sv_file,os.path.join(save_path,'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val ==0):
            val_res_mae,val_res_rmse,val_res_mb,val_res_cc=eval_mae_psnr_rmse_mb_cc(model,data_dict,
                              eval_bsize=config.get('eval_bsize'))

            log_info.append('val: mae={:.4f},rmse={:.4f},mb={:.4f},cc={:.4f}'.format(val_res_mae,val_res_rmse,val_res_mb,val_res_cc))
            writer.add_scalars('mae',{'val':val_res_mae},epoch)
            writer.add_scalars('mb',{'val':val_res_mb},epoch)
            writer.add_scalars('rmse',{'val':val_res_rmse},epoch)
            writer.add_scalars('cc',{'val':val_res_cc},epoch)
            if val_res_mae<min_val_v:
                min_val_v=val_res_mae
                torch.save(sv_file,os.path.join(save_path,'epoch-best.pth'))
        t=timer.t()
        prog=(epoch-epoch_start+1)/(epoch_max-epoch_start+1)
        t_epoch=utils.time_text(t-t_epoch_start)
        t_elapsed,t_all=utils.time_text(t),utils.time_text(t/prog)
        log_info.append('{} {}/{}'.format(t_epoch,t_elapsed,t_all))
        log(' '.join(log_info))
        writer.flush()

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


if __name__ == '__main__':
    # 纬度范围: 37.49111 ~ 42.587223
    # 经度范围: 113.61361~ 119.288055

    #grid最小值为: 241.14527893066406
    #大值为: 300.359130859375

    #obs最小值: -17.0
    #最大值: 22.0
    file_path_gridandcontext = r'/mnt/nas/b103/huxiaochuan/newmethon_data/caijian_grid'
    file_path_obseg = r'/mnt/nas/b103/huxiaochuan/newmethon_data/suiji_xunliandata'
    data_dict = create_file_list(file_path_gridandcontext)

    all_data = []
    num=0
    for task in data_dict:
        file_name = os.path.join(file_path_gridandcontext, task)
        file_name_obseg = os.path.join(file_path_obseg, task)

        contextgrid_file = os.path.join(file_name, task + '_contextgrid.npy')
        target_file = os.path.join(file_name_obseg, task + 'target.pt')
        obs_file = os.path.join(file_name_obseg, task + 'obs.npy')

        grid = load_data_from_file(contextgrid_file)
        target = load_data_from_file(target_file)
        gt = load_data_from_file(obs_file)

        all_data.append((grid, target, gt))
        num=num+1
        print(f"第{num}条数据加载完成")


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/mnt/hdd1/huxiaochuan/AMSD_zhandian/configs/train-meso/train_amsd-baseline-lte.yaml')
    parser.add_argument('--name', default='meso')
    parser.add_argument('--tag', default='v4')
    parser.add_argument('--gpu', default='2')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_'+args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_'+args.tag
    save_path=os.path.join('/mnt/nas/b103/huxiaochuan/AMSD_zhandian',save_name)

    main(config,save_path)

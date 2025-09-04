import argparse
import os
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
from test import eval_mae_psnr_rmse_mb_cc_ssim

def make_data_loader(spec,tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset=datasets.make(spec['wrapper'],args={'dataset':dataset})

    log(rf'{tag} dataset: size={len(dataset)}')
    for k,v in dataset[0].items():
        log(rf'{k}:shape={tuple(v.shape)}')

    loader = DataLoader(dataset,batch_size=spec['batch_size'],
                        shuffle=(tag=='train'),num_workers=0,pin_memory=True)
    return loader

def make_data_loaders():
    train_loader=make_data_loader(config.get('train_dataset'),tag='train')
    val_loader=make_data_loader(config.get('val_dataset'),tag='val')
    return train_loader,val_loader

def prepare_training():
    if os.path.exists(config.get('resume')):
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'],load_sd=True).cuda()
        optimizer= utils.make_optimizer(model.parameters(),sv_file['optimizer'])
        epoch_start=sv_file['epoch']+1
        if config.get('multi_step_lr') is None:
            lr_scheduler=None
        else:
            lr_scheduler= MultiStepLR(optimizer,**config['multi_step_lr'])
        for _ in range(epoch_start -1):
            lr_scheduler.step()
    else:
        model=models.make(config['model']).cuda()
        optimizer=utils.make_optimizer(model.parameters(),config['optimizer'])
        epoch_start=1
        if config.get('multi_step_lr') is None:
            lr_scheduler=None
        else:
            lr_scheduler= MultiStepLR(optimizer,**config['multi_step_lr'])
    log('model: #params={}'.format(utils.compute_num_params(model,text=True)))
    return model,optimizer,epoch_start,lr_scheduler

def train(train_loader,model,optimizer,epoch):
    model.train()
    loss_fn=nn.L1Loss()
    train_loss=utils.Averager()

    num_dataset=1000
    iter_per_epoch=int(num_dataset/config.get('train_dataset')['batch_size'])
    iteration=0
    for batch in tqdm(train_loader,leave=False,desc='train'):
        for k,v in batch.items():
            batch[k]=v.cuda()
        pred=model(batch['inp'],batch['coord'],batch['cell'])
        gt=batch['gt']
        loss=loss_fn(pred,gt)
        writer.add_scalars('loss',{'train':loss.item()},(epoch-1)*iter_per_epoch+iteration)
        iteration+=1

        train_loss.add(loss.item(),gt.shape[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss.item()

def main(config_,save_path):
    global config,log,writer
    config=config_
    log,writer=utils.set_save_path(save_path,remove=False)
    with open(os.path.join(save_path,'config.yaml'),'w') as f:
        yaml.dump(config,f,sort_keys=False)

    train_loader,val_loader=make_data_loaders()

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
        train_loss=train(train_loader,model,optimizer,epoch)
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
            val_res_mae,val_res_rmse,val_res_spearm,val_res_std,val_res_mb,val_res_cc,val_res_p98=eval_mae_psnr_rmse_mb_cc_ssim(val_loader,model,
                              eval_bsize=config.get('eval_bsize'))

            log_info.append('val: mae={:.4f}, psnr={:.4f},rmse={:.4f},mb={:.4f},cc={:.4f},ssim={:.4f}'.format(val_res_mae,val_res_rmse,val_res_mb,val_res_spearm,val_res_cc,val_res_std,val_res_p98))
            writer.add_scalars('mae',{'val':val_res_mae},epoch)
            writer.add_scalars('spearm',{'val':val_res_spearm},epoch)
            writer.add_scalars('mb',{'val':val_res_mb},epoch)
            writer.add_scalars('rmse',{'val':val_res_rmse},epoch)
            writer.add_scalars('cc',{'val':val_res_cc},epoch)
            writer.add_scalars('std',{'val':val_res_std},epoch)
            writer.add_scalars('p98',{'val':val_res_p98},epoch)

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/mnt/hdd1/huxiaochuan/AMSD_wangge/configs/train-meso/train_ours.yaml')
    parser.add_argument('--name', default='BY3')
    parser.add_argument('--tag', default='ours_v1')
    parser.add_argument('--gpu', default='1')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_'+args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_'+args.tag
    save_path=os.path.join('/mnt/nas/b103/huxiaochuan/AMSD_lteandduomokuan_xiaorong/',save_name)

    main(config,save_path)

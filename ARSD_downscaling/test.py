import argparse
import os
import math
from functools import partial
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
import models
import utils
import numpy as np
from glob import glob
from models import make
def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql:qr, :], cell[:, ql:qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred

def eval_mae_psnr_rmse_mb_cc_ssim(loader, model, eval_bsize=None, verbose=False):
    model.eval()
    val_res_mae = utils.Averager()
    val_res_std = utils.Averager()
    val_res_biasp98 = utils.Averager()
    val_res_rmse = utils.Averager()
    val_res_mb = utils.Averager()
    val_res_pearson = utils.Averager()
    val_res_spearman = utils.Averager()
    pbar = tqdm(loader, leave=False, desc='val')
    count=0
    first_batch_processed = False
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()
        inp, coord, cell = batch['inp'], batch['coord'], batch['cell']
        with torch.no_grad():
            if eval_bsize is None:
                pred = model(inp, coord, cell)
            else:
                pred = batched_predict(model, inp, coord, cell, eval_bsize)
        pred.clamp_(0., 1.)
        t_pred=pred
        count=count+1
        sr, hr = t_pred[0, :, 0].detach().cpu().numpy(), batch['gt'][0, :, 0].detach().cpu().numpy()
        res_mae, res_rmse, res_mb, res_speram, res_pearson, res_std,bias_p98 = utils.calc_mae_rmse_mb_cc_psnr_ssim(sr, hr)
        val_res_mae.add(res_mae.item(), inp.shape[0])
        val_res_rmse.add(res_rmse.item(), inp.shape[0])
        val_res_spearman.add(res_speram.item(), inp.shape[0])
        val_res_std.add(res_std.item(), inp.shape[0])
        val_res_mb.add(res_mb.item(), inp.shape[0])
        val_res_pearson.add(res_pearson.item(), inp.shape[0])
        val_res_biasp98.add(bias_p98.item(), inp.shape[0])
    return val_res_mae.item(), val_res_rmse.item(), val_res_spearman.item(),val_res_std.item(),val_res_mb.item(),val_res_pearson.item(),val_res_biasp98.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/mnt/hdd1/huxiaochuan/AMSD_wangge/configs/test/x16test_ours.yaml')
    #parser.add_argument('--model', type=str, default='/mnt/nas/b103/huxiaochuan/AMSD_wangge_duomokuan/BY3_ours_v1/epoch-20.pth')#Ours
    parser.add_argument('--gpu', type=str, default='2')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=0, shuffle=False, pin_memory=True)
    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    res_mae, res_rmse, res_spearman, res_std, res_mb,res_pearson,res_biasp98 = eval_mae_psnr_rmse_mb_cc_ssim(loader, model,
                                                                           eval_bsize=config.get('eval_bsize'),
                                                                           verbose=True)
    utils.log(
        'result mae: {:.4f},rmse: {:.4f}, spearman: {:.4f}, std: {:.4f}, mb: {:.4f}, pearson:{:.4f}, biasp98:{:.4f}'.format(res_mae, res_rmse, res_spearman,
                                                                                        res_std, res_mb, res_pearson,res_biasp98))


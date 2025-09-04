from datasets import register
from torch.utils.data import Dataset
import numpy as np
import os
import utils
@register('image-folder')
class ImageFolder(Dataset):
    def __init__(self, root_path):
        self.static=np.load(r'/mnt/hdd1/zhanghonghu1/DATA/MeteoNet_2018/data_NW/Land-Sea and Relief Masks/lat_lon_lsm_h.npz')
        paths = sorted(os.listdir(root_path[2:]))
        if root_path[:2]=='Tr':
            self.paths = [os.path.join(root_path[2:], path) for path in paths[:1000]]
            self.ids_inp = np.load(r'/mnt/hdd1/zhanghonghu1/DATA/MeteoNet_2018/站点分布图_train_200.npz')['id']
            self.ids_gt = None
        else:
            self.paths=[os.path.join(root_path[2:], path) for path in paths[1000:1100]]
            self.ids_inp = np.load(r'/mnt/hdd1/zhanghonghu1/DATA/MeteoNet_2018/站点分布图_train_200.npz')['id']
            self.ids_gt = np.load(r'/mnt/hdd1/zhanghonghu1/DATA/MeteoNet_2018/站点分布图_test_60.npz')['id']
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        x=np.load(self.paths[idx],allow_pickle=True)

        return {
            'x': x,
            'ids_inp': self.ids_inp,
            'ids_gt': self.ids_gt,
            'static': self.static
        }

if __name__=='__main__':
    import argparse

    #test1
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default=r'Tr/mnt/hdd1/zhanghonghu1/DATA/MeteoNet_2018/v5_nor',
                        help='path to dataset')
    args = parser.parse_args()
    dataset=ImageFolder(**vars(args))

    print(len(dataset))
    # print(dataset[0].shape)
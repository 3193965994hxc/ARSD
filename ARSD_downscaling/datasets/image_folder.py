from datasets import register
from torch.utils.data import Dataset
import numpy as np

@register('image-folder')
class ImageFolder(Dataset):
    def __init__(self, root_path):
        if root_path[:2]=='Tr':
            self.data=np.load(root_path[2:])['data'][:1000,:,:]
        else:
            self.data = np.load(root_path[2:])['data'][1036:1037, :, :] #[1036:1037, :, :]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        y=self.data[idx]
        return y

if __name__=='__main__':
    import argparse

    #test1
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default=r'Tr/mnt/nas/b103/zhanghonghu1/data/MESO/grapes_meso_3km_T2m_1200_13_653_min_max_nor.npz',
                        help='path to dataset')
    args = parser.parse_args()
    dataset=ImageFolder(**vars(args))

    print(len(dataset))
    # print(dataset[0].shape)
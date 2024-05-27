
import os

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from glob import glob
from helpers.helpers import *

def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def matpadder(x, max_in=512):
    if len(x.shape) < 2:
        x = x.unsqueeze(0)
    shape = x.shape
    # delta1 = max_in - shape[0]
    delta2 = max_in - shape[-1]

    out = F.pad(x, (0, delta2, 0, 0), "constant", 0)
    return out

class ZooDataset(Dataset):
    """weights dataset."""
    def __init__(self, root='zoodata', dataset='joint', split='train', scale=0.125, topk=None, transform=None, normalize=False,
                 max_len=11689512):
        super(ZooDataset, self).__init__()
        self.dataset = dataset
        self.topk=topk
        self.max_len = max_len
        self.normalize = normalize
        self.scale = scale
        # root = '../../../Datasets/imnet20kFeatures'

        datapath = '../Datasets/nndzoo/minizoo/checkpoints/'



        self.transform = transform
        files_list = self.load_data(datapath)

        print(len(files_list))

        self.files_list = files_list

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fls = self.files_list[idx]
        std = torch.load(fls , map_location='cpu')
        weight = gets_weights(std)
        if weight.shape[-1] < self.max_len:
            weight = matpadder(weight, self.max_len)
        weight= weight/self.scale
        weight = weight.reshape(-1)
        return weight
    def load_data(self, root_dir):

        dsets = list(os.listdir(root_dir))
        files =[]
        for dset in dsets:
            flrs =  glob(f'{root_dir}/{dset}/whole_model_*_{dset}.pth')
            files += flrs
        del dsets
        return files



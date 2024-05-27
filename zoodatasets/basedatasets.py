
import os

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset

def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def matpadder(x, max_in=512):
    shape =x.shape
    # delta1 = max_in - shape[0]
    delta2 = max_in - shape[1]

    out = F.pad(x, (0, delta2, 0, 0), "constant", 0)
    return out

class ZooDataset(Dataset):
    """weights dataset."""
    def __init__(self, root='zoodata', dataset='joint', split='train', scale=1.0, topk=None, transform=None, normalize=False,
                 max_len=2864):
        super(ZooDataset, self).__init__()
        self.dataset = dataset
        self.topk=topk

        self.max_len = max_len
        self.normalize = normalize
        self.split=split
        self.scale= scale

        datapath = os.path.join(root, f'weights/{split}_data.pt')


        self.transform = transform
        data = self.load_data(datapath)

        print(data.shape)

        self.data = data/self.scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        weight = self.data[idx].to(torch.float32)
        if self.transform:
            weight = self.transform(weight)
        sample = {'weight': weight, 'dataset': []}
        return  sample
    def load_data(self, file): #loading pretrained weights vector from dict {datset:weights)
        data = torch.load(file)
        wl = []
        keys = list(data)

        for k in keys:
            w = data[k][0].detach().cpu()
            # print(w.shape)
            if len(w.shape)<2:
                w = w.unsqueeze(0)
            if w.shape[-1] < self.max_len:
                w = matpadder(w, self.max_len)

            if self.topk is not None:
                # print(w)
                wl.append(w[:self.topk])
            else:
                wl.append(w)

        data = torch.cat(wl, dim=0)


        return data



import os

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
import collections
from glob import glob



def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def matpadder(x, max_in=512):
    shape = x.shape
    # delta1 = max_in - shape[0]
    delta2 = max_in - shape[1]

    out = F.pad(x, (0, delta2, 0, 0), "constant", 0)
    return out

class ZooDataset(Dataset):
    """weights dataset."""

    def __init__(self, root='zoodata', dataset='joint', split='train', scale=1.0, num_sample=5, topk=None, transform=None,
                 normalize=False,
                 max_len=2864):
        super(ZooDataset, self).__init__()
        self.dataset = dataset
        self.topk = topk
        self.max_len = max_len
        self.split = split
        self.normalize = normalize
        self.num_sample = num_sample
        self.scale = scale
        self.root = root
        datapath = os.path.join(root, f'weights/{split}_data')


        self.transform = transform

        self.data = self.load_data(datapath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        weights = self.data[idx]
        # targets = self.targets[idx]
        # flrs = os.listdir("../Datasets/imnet1kzoo/condata")
        wl = []
        conds = []
        keys = list(weights)
        for k in keys:
            w = weights[k]/self.scale
            if w.shape[1] < self.max_len:
                w = matpadder(w, self.max_len)
            x = torch.load(f"../Datasets/imnet1kzoo/res50_mixe4d_20k_condata/{k}/train_cond_.pt")
            # x = targets[k]
            # cdata = []
            num_class = len(x)
            classes = list(range(num_class))
            # print(num_class)
            xd = []
            # enabling sampling new batch of image during training
            for i in range(w.shape[0]):
                cdata = []
                for cls in classes:
                    cx = x[cls]
                    ridx = torch.randperm(len(cx))
                    cdata.append(cx[ridx][:self.num_sample])
                # xd.append(torch.stack(cdata, 0))
                conds.append(torch.stack(cdata, 0).type(torch.float32))
            wl.append(w)
        target = conds
        weight = w
        sample = {'weight': weight, 'dataset': target}
        del x
        return sample

    def load_data(self, file):
        data = torch.load(file)
        # xc = torch.load('../Datasets/vitzoo/conds/clip_dsets_train_40_conds_vit_.pt')
        # xc = torch.load('../Datasets/imnet21kzoo/swint/for_clip_20_samples_swint_train_.pt')
        # flrs = os.listdir("../Datasets/imnet1kzoo/condata")
        xdata = []
        cond = []
        ydata = []
        # weights ={}
        # y = {}
        keys = list(data)
        for k in keys:
            # if k not in toreomve:

            w = data[k][0]

            w = w.detach().cpu()
            if len(w.shape)<2:
                w = w.unsqueeze(0)
            # for i in range(w.shape[0]):
            xdata.append({k: w})
            # cond.append({k: xc[k]})
            # ydata.append({k: y})

        # xdata = [{k:v} for k,v in data.items()]
        # ydata = [{k:v} for k,v in xc.items()]

        return xdata




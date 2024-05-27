import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
#

from zoodatasets.sampler import ZooDataset


def my_collate(batch):
    sample = {}
    data = [item['weight'][0] for item in batch]
    conds = [item['dataset'] for item in batch]
    target =[]
    for items in conds:
        target += [item for item in items]
    data = torch.cat(data, 0)
    sample['weight']=data
    sample['dataset'] = target

    return sample

def m_collate(batch):
    sample = {}
    data = [item['weight'] for item in batch]
    target = [item['dataset'] for item in batch]
    data = torch.stack(data, 0).type(torch.float32)
    sample['weight']=data.type(torch.float32)
    sample['dataset'] = target
    # exit()
    return sample



class ZooDataModule(pl.LightningDataModule):
    def __init__(self, dataset, data_dir, data_root, batch_size, num_workers, num_sample, topk, normalize, scale=1.0):
        super().__init__()
        self.data_dir = data_dir
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.topk = topk
        self.normalize = normalize
        self.num_sample = num_sample
        self.scale= scale

        # self.transform = []

    def prepare_data(self):
        datasets.CIFAR10(self.data_root, train=True, download=True)
        datasets.CIFAR10(self.data_root, train=False, download=True)

    def setup(self, stage):

        if stage == "fit":
            self.trainset = ZooDataset(root=self.data_dir, dataset=self.dataset, split='train', scale=self.scale,
                                       topk=self.topk, num_sample=self.num_sample)
            self.valset = ZooDataset(root=self.data_dir, dataset=self.dataset, split='val', scale=self.scale,
                                     num_sample=self.num_sample)

        if stage == "test":
            self.testset = ZooDataset(root=self.data_dir, dataset=self.dataset, split='val', scale=self.scale,
                                      num_sample=self.num_sample)

        if stage == "predict":
            # self.cifar10_predict = datasets.CIFAR10(self.data_root, train=False, transform=self.transform)
            pass

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=my_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=my_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=m_collate,
        )

    # dataloader to evaluate the reconstruction performance on model zoo.
    # def predict_dataloader(self):
    #     return DataLoader(
    #         self.cifar10_predict,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #     )


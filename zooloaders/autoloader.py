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


from zoodatasets.basedatasets import ZooDataset



class ZooDataModule(pl.LightningDataModule):
    def __init__(self, dataset, data_dir, data_root, batch_size, num_workers, scale, num_sample, topk, normalize):
        super().__init__()
        self.data_dir = data_dir
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.topk = topk
        self.normalize = normalize
        self.num_sample = num_sample
        self.scale=scale


        # self.transform = []

    def prepare_data(self):
        datasets.CIFAR10(self.data_root, train=True, download=True)
        datasets.CIFAR10(self.data_root, train=False, download=True)

    def setup(self, stage):

        if stage == "fit":
            self.trainset = ZooDataset(root=self.data_dir, dataset=self.dataset, split='train', scale=self.scale, topk=self.topk)
            self.valset = ZooDataset(root=self.data_dir, dataset=self.dataset, split='val',scale=self.scale)

        if stage == "test":
            self.testset = ZooDataset(root=self.data_dir, dataset=self.dataset, split='test', scale=self.scale)

        if stage == "predict":
            pass
            # pass

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    # dataloader to evaluate the reconstruction performance on model zoo.
    def predict_dataloader(self):
        pass
        # return DataLoader(
        #     self.cifar10_predict,
        #     batch_size=self.batch_size,
        #     num_workers=self.num_workers,
        #     shuffle=False,
        # )



import numpy as np
import torchvision.models as models
import torchvision.datasets as dset
import os
import torch
import argparse
import random
import torchvision.transforms as transforms
import os, sys
from torchvision.models import resnet18, ResNet18_Weights

import pickle
from PIL import Image

parser = argparse.ArgumentParser("sota")
parser.add_argument('--gpu', type=str, default='0', help='set visible gpus')

parser.add_argument('--save_path', type=str, default='imgdata', help='the path of save directory')
parser.add_argument('--dataset', type=str, default='cifar10-1', help='choose dataset')
parser.add_argument('--data_path', type=str, default='../Datasets/data', help='the path of save directory')
parser.add_argument('--seed', type=int, default=23, help='random seed')
args = parser.parse_args()

if args.seed is None or args.seed < 0: args.seed = random.randint(1, 100000)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
np.random.seed(args.seed)
random.seed(args.seed)

# remove last fully-connected layer
# model = models.resnet18(pretrained=True).eval()
# model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

def get_transform(dataset):

    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    if dataset == 'mnist':
        transform.transforms.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    transform.transforms.append( transforms.Normalize(mean, std))
    return transform


def process(dataset, n_classes):
    data_label = {i: [] for i in range(n_classes)}
    for x, y in dataset:
        data_label[y].append(x)
    for i in range(n_classes):
        data_label[i] = torch.stack(data_label[i])

    holder = {}
    for i in range(n_classes):
        with torch.no_grad():
            xd = data_label[i][:20]
            data = model.encode_image(xd.to(device))
            data = data.squeeze()

            holder[i] = data.detach().cpu().type(torch.float16)

    return holder


import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import clip
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    cdata = {}

    #Tiny_subsets
    #Tinycent_subsets
    # ncls = {'mnist': 10, 'svhn': 10, 'cifar10': 10, 'cifar100': 100,  'aircraft30': 30, 'aircraft100': 100, 'pets': 37}
    # ncls = {'mnist': 10, 'svhn': 10, 'cifar10': 10, 'stl10': 10, 'cifar100': 100, 'aircraft30':30, 'pets':37}
    # ncls = {'svhn': 10, 'cifar10': 10, 'stl10': 10, 'cifar100': 100, 'aircraft30': 30, 'pets': 37}
    ncls = {'mnist': 10, 'svhn': 10, 'cifar10': 10, 'stl10': 10}
    dsets = list(ncls)
    for dst in tqdm(dsets):
        args.dataset= dst
        transform = get_transform(dst)
        # dset = i
        if args.dataset == 'mnist':
            data = dset.MNIST(args.data_path, train=True, transform=transform, download=True)

        if args.dataset == 'fashionmnist':
            data = dset.FashionMNIST(args.data_path, train=True, transform=transform, download=True)

        elif args.dataset == 'svhn':
            data = dset.SVHN(args.data_path, split='train', transform=preprocess , download=True)
        elif args.dataset == 'cifar10':
            data = dset.CIFAR10(args.data_path, train=True, transform=preprocess, download=True)

        # elif args.dataset == 'cifar100':
        #     data = dset.CIFAR100(args.data_path, train=False, transform=transform, download=True)
            # print(data)
        elif args.dataset == 'stl10':
            data = dset.STL10(args.data_path, split='train', transform=preprocess, download=True)

        elif args.dataset == 'cifar100':
            data = dset.CIFAR100(args.data_path, train=False, transform=preprocess, download=True)

        elif args.dataset == 'aircraft30':
            data = dset.FGVCAircraft(root=args.data_path, split='test', transform=preprocess, annotation_level='manufacturer',
                                      download=True)

        elif args.dataset == 'aircraft100':
            data = dset.FGVCAircraft(root=args.data_path, split='test', transform=preprocess, annotation_level='variant',
                                     download=True)

        elif args.dataset == 'pets':
            data = dset.OxfordIIITPet(args.data_path, split='test', transform=preprocess, target_types='category',
                                       download=True)


        if args.dataset == 'emnist':
            data = dset.EMNIST(args.data_path, split='digits', train=False, transform=preprocess, download=True)

        dataset = process(data, ncls[dst])
        cdata[dst]= dataset

    torch.save(cdata, 'clip_encode_dsets_20_cond_.pt')

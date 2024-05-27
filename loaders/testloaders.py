
import os
import torch
import torchvision

from timm.data.transforms import str_to_interp_mode

from timm.data.auto_augment import rand_augment_transform

import torch.utils.data

import torchvision.transforms as transforms
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import math



def _convert_image_to_rgb(image):
    return image.convert("RGB")

def build_train_transform(image_size=32, auto_augment='rand-m9-mstd0.5'):
    img_size_min = image_size

    train_transforms = [transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        ]

    aa_params = dict(
        translate_const=int(img_size_min * 0.45),
        img_mean=tuple([min(255, round(255 * x)) for x in [0.48933587508932375, 0.5183537408957618,
                                                           0.5387914411673883]]),
    )
    aa_params['interpolation'] = _str_to_pil_interpolation('bicubic')
    train_transforms += [rand_augment_transform(auto_augment, aa_params)]

    train_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48933587508932375, 0.5183537408957618, 0.5387914411673883],
            std=[0.22388883112804625, 0.21641635409388751, 0.24615605842636115]),
    ]

    train_transforms = transforms.Compose(train_transforms)
    return train_transforms

mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]



def get_transform(dataset):
    if dataset == 'mnist':
        mean, std = (0.1307), (0.3081)
    elif dataset == 'svhn':
        mean, std = [0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614]
    elif dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = [0.2023, 0.1994, 0.2010]
    elif dataset == 'cifar100':
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    elif dataset=='aircraft30':
        mean = [0.48933587508932375, 0.5183537408957618, 0.5387914411673883]
        std = [0.22388883112804625, 0.21641635409388751, 0.24615605842636115]
    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]







def get_loader(dataset='mnist', batch_size=512):
    if dataset == 'mnist':
        mean, std = (0.1307), (0.3081)
    elif dataset == 'svhn':
        mean, std = [0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614]
    elif dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = [0.2023, 0.1994, 0.2010]
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
    elif dataset == 'cifar100':
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    elif dataset == 'aircraft30':
        mean = [0.48933587508932375, 0.5183537408957618, 0.5387914411673883]
        std = [0.22388883112804625, 0.21641635409388751, 0.24615605842636115]
    elif dataset == 'imagenet':
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    data_path = '../Datasets/data'
    # train_transform, test_transform = get_transform(dataset)
    # ncls = {'mnist': 10, 'svhn': 10, 'cifar10': 10, 'stl10': 10,  'aircraft30': 30, 'aircraft100': 100, 'pets': 37}

    if dataset == 'mnist':
        train_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean, std), ])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean, std),])
        train_data = dset.MNIST(data_path, train=True, transform=train_transform, download=True)
        test_data = dset.MNIST(data_path, train=False, transform=test_transform, download=True)
        trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size,
                                                  num_workers=4, pin_memory=True, shuffle=True)
        valloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size,
                                                shuffle=False)
        return trainloader, valloader

    if dataset == 'fashionmnist':
        train_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean, std), ])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean, std), ])
        train_data = dset.FashionMNIST(data_path, train=True, transform=train_transform, download=True)
        test_data = dset.FashionMNIST(data_path, train=False, transform=test_transform, download=True)
        trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size,
                                                  num_workers=4, pin_memory=True, shuffle=True)
        valloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size,
                                                shuffle=False)
        return trainloader, valloader

    elif dataset == 'svhn':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std), ])
        test_transform = transforms.Compose([transforms.Resize((32,32)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std), ])
        train_data = dset.SVHN(data_path, split='train', transform=train_transform, download=True)
        test_data = dset.SVHN(data_path, split='test', transform=test_transform, download=True)
        trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size,
                                                  num_workers=4, pin_memory=True, shuffle=True)
        valloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size,
                                                shuffle=False)
        return trainloader, valloader
    elif dataset == 'cifar10':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std), ])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean, std), ])
        train_data = dset.CIFAR10(data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(data_path, train=False, transform=test_transform, download=True)
        trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size,
                                                  num_workers=4, pin_memory=True, shuffle=True)
        valloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size,
                                                shuffle=False)
        return trainloader, valloader
        # print(data)
    elif dataset == 'stl10':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std), ])
        test_transform = transforms.Compose([transforms.Resize((32,32)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std), ])
        train_data = dset.STL10(data_path, split='train', transform=train_transform, download=True)
        test_data = dset.STL10(data_path, split='test', transform=test_transform, download=True)
        trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size,
                                                  num_workers=4, pin_memory=True, shuffle=True)
        valloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size,
                                                shuffle=False)
        return trainloader, valloader

    elif dataset == 'cifar100':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std), ])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean, std), ])
        train_data = dset.CIFAR100(data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(data_path, train=False, transform=test_transform, download=True)
        trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size,
                                                  num_workers=4, pin_memory=True, shuffle=True)
        valloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size,
                                                shuffle=False)
        return trainloader, valloader

    elif dataset == 'aircraft30':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std), ])
        test_transform = transforms.Compose([transforms.Resize((32,32)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std), ])
        # train_transform= build_train_transform()
        train_data = dset.FGVCAircraft(root=data_path, split='train', transform=train_transform, annotation_level='manufacturer',
                                  download=True)
        test_data = dset.FGVCAircraft(root=data_path, split='test', transform=test_transform,
                                 annotation_level='manufacturer',
                                 download=True)
        trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size,
                                                  num_workers=4, pin_memory=True, shuffle=True)
        valloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size,
                                                shuffle=False)
        return trainloader, valloader

    elif dataset == 'aircraft100':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std), ])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean, std), ])
        train_data = dset.FGVCAircraft(root=data_path, split='train', transform=train_transform, annotation_level='variant',
                                 download=True)
        test_data = dset.FGVCAircraft(root=data_path, split='test', transform=test_transform, annotation_level='variant',
                                 download=True)
        trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size,
                                                  num_workers=4, pin_memory=True, shuffle=True)
        valloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, num_workers=4,
                                                shuffle=False)
        return trainloader, valloader

    elif dataset == 'pets':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std), ])
        test_transform = transforms.Compose([transforms.Resize((32,32)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std), ])
        train_data = dset.OxfordIIITPet(data_path, split='trainval', transform=train_transform, target_types='category',
                                   download=True)
        test_data = dset.OxfordIIITPet(data_path, split='test', transform=test_transform, target_types='category',
                                  download=True)
        trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size,
                                                  num_workers=4, pin_memory=True, shuffle=True)
        valloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size,
                                                shuffle=False)
        return trainloader, valloader

    elif dataset == 'imagenet':
        data_path = '../Datasets/imagenet'

        traindir = os.path.join(data_path, 'train')
        valdir = os.path.join(data_path, 'val')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])


        train_dataset = dset.ImageFolder(
            traindir,
            transforms.Compose([
                _convert_image_to_rgb,
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = dset.ImageFolder(
            valdir,
            transforms.Compose([
                _convert_image_to_rgb,
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers= 4)
        return train_loader, val_loader

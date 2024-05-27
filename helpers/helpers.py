import argparse
import os
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import yaml

def add_to_config(mydict, cfl):
    with open(cfl, 'a') as configfile:
        data = yaml.dump(mydict, configfile, indent=4)
        print("Write successful")

def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def set_state_dict(std, weights):
    # std = model.state_dict()
    st = 0
    for params in std:
        if not params.endswith('num_batches_tracked'):
            shape = std[params].shape
            ed = st + np.prod(shape)
            std[params] = weights[st:ed].reshape(shape)
            # model.load_state_dict(std)
            st = ed
    return std

def gets_weights(std):
    # std = model.state_dict()
    weights = []
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if 'mean' in params or 'var' in params:
                continue
            # print(params)
            w = std[params].reshape(-1)
            weights.append(w)
    return torch.cat(weights, -1)


def set_model_weights(model, weights):
    std = model.state_dict()
    st = 0
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if params.endswith('running_var') or params.endswith('running_mean'):
                continue
            # elif 'linear' in params:
            #     continue
            shape = std[params].shape
            ed = st + np.prod(shape)
            std[params] = weights[st:ed].reshape(shape)
            model.load_state_dict(std)
            st = ed
    return model



def set_weights(model, weights):
    std = model.state_dict()
    st = 0
    for params in std:
        if not params.endswith('num_batches_tracked'):
            shape = std[params].shape
            ed = st + np.prod(shape)
            std[params] = weights[st:ed].reshape(shape)
            model.load_state_dict(std)
            st = ed



def vecpadder(x, max_in=3728761 * 3):
    shape = x.shape
    delta1 = max_in - shape[0]
    x = F.pad(x, (0, delta1))
    return x


def pad_to_chunk_multiple(x, chunk_size):
    shape = x.shape
    if len(shape)<2:
        x =x.unsqueeze(0)
        shape = x.shape
    max_in = chunk_size*math.ceil(shape[1]/chunk_size)
    delta1 = max_in - shape[1]
    # x = F.pad(x, (0, delta1))
    x =F.pad(x, (0, delta1, 0, 0), "constant", 0)
    return x

def matpadder(x, max_in=512):
    shape =x.shape
    # delta1 = max_in - shape[0]
    delta2 = max_in - shape[1]

    out = F.pad(x, (0, delta2, 0, 0), "constant", 0)
    return out
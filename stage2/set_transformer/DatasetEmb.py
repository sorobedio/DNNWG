import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils.util import  instantiate_from_config
from stage2.set_transformer.models import SetTransformer
from stage2.set_transformer.super_linear import *






class IdentityCondStage(torch.nn.Module):
    def __init__(self, in_channels, input_size, **kwargs):
        super().__init__()
        self.in_channels =in_channels
        self.input_size = input_size

    def forward(self, x, *args, **kwargs):
        # x = x.reshape((-1, 5, 32, 32))
        return x


class MyMLPEncoder(nn.Module):
    def __init__(self, in_ch=1, num_samples=5, input_size=32, num_classes=10, out_dim=4, embed_dim=512, **kwargs):
        super(MyMLPEncoder, self).__init__()
        self.in_ch = in_ch
        self.num_sample=num_samples
        self.n_classes = num_classes
        self.max_classes = num_classes
        self.max_dim = num_samples*num_classes
        self.in_res = input_size
        self.out_dim = out_dim
        self.embed_dim= embed_dim
        infeat = embed_dim*out_dim
        self.dense1 = LinearSuper(super_in_dim=num_samples*num_classes, super_out_dim=out_dim)
        self.dense2 = nn.Linear(infeat, in_ch*input_size*input_size)
        # self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        out =[]
        for x in inputs:
            # print(len(x))
            if isinstance(x, list):
                if len(x)==1:
                    x= x[0]
                # x = torch.stack(x, dim=0)
            assert len(x.shape)==3, "x  should have 3 dimensions"
            ns = x.shape[1]
            nc = x.shape[0]
            dim = nc*ns
            if dim > self.max_dim :
                dim = self.max_dim
                x = x[:self.max_classes]
            # else:
            self.dense1.set_sample_config(dim, self.out_dim)

            x = x.cuda()
            x = rearrange(x, 'c n d -> d (n c)')
            x = self.dense1(x)
            x = F.leaky_relu(x)
            out.append(x)
        out = torch.stack(out, 0)
        x = rearrange(out, 'b d n -> b (d n)')
        x = self.dense2(x)
        x = x.reshape(-1, self.in_ch, self.in_res, self.in_res)
        return x



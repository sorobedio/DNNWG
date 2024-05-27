import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils.util import  instantiate_from_config
from stage2.set_transformer.models import SetTransformer


class IdentityCondStage(torch.nn.Module):
    def __init__(self, in_channels, input_size, **kwargs):
        super().__init__()
        self.in_channels =in_channels
        self.input_size = input_size

    def forward(self, x, *args, **kwargs):
        # x = x.reshape((-1, 5, 32, 32))
        return x



class EmbedData(nn.Module):
    def __init__(self,  enconfig, deconfig, emb_dim=1024,  **kwargs):
        super(EmbedData, self).__init__()
        self.intra = SetTransformer(enconfig, deconfig)
        self.inter = SetTransformer(enconfig, deconfig)
        self.emb_dim = emb_dim
        # self.input_size= input_size
        # self.channels = channels
        self.proj = nn.Linear(512, emb_dim)

    def forward(self, inputs):
        outputs = []
        for x in inputs:
            if isinstance(x, list) and len(x)>=1:
                x = torch.stack(x, dim=0)
            elif isinstance(x, list) and len(x)==1:
                x= x[0]

            x =x.cuda()
            z = self.intra(x).squeeze(1)
            z = z.unsqueeze(0)
            out = self.inter(z).reshape(-1)
            outputs.append(out)
        outputs = torch.stack(outputs, 0).reshape(-1, 512)
        outputs = self.proj(outputs)
        return outputs

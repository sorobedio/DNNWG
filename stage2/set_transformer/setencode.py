###########################################################################################
# Copyright (c) Hayeon Lee, Eunyoung Hyung [GitHub MetaD2A], 2021
# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021
###########################################################################################
import torch

from stage2.set_transformer.modules import *


class SetPool(nn.Module):
  def __init__(self, dim_input, num_outputs, dim_output,
        num_inds=32, dim_hidden=128, num_heads=4, ln=False, mode=None):
    super(SetPool, self).__init__()

    if 'sab' in mode: # [32, 400, 128]
      self.enc = nn.Sequential(
        SAB(dim_input, dim_hidden, num_heads, ln=ln),  # SAB?
        SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
    else: # [32, 400, 128]
      self.enc = nn.Sequential(
        ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),  # SAB?
        ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
    if 'PF' in mode: #[32, 1, 501]
      self.dec = nn.Sequential(
        PMA(dim_hidden, num_heads, num_outputs, ln=ln),
        nn.Linear(dim_hidden, dim_output))
    elif 'P' in mode:
      self.dec = nn.Sequential(
        PMA(dim_hidden, num_heads, num_outputs, ln=ln))
    else: #torch.Size([32, 1, 501])
      self.dec = nn.Sequential(
        PMA(dim_hidden, num_heads, num_outputs, ln=ln), # 32 1 128
        SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        nn.Linear(dim_hidden, dim_output))
  # "", sm, sab, sabsm
  def forward(self, X):
    x1 = self.enc(X)
    x2 = self.dec(x1)
    return x2
#############################################################################
class SetAgregate(nn.Module):
    def __init__(self, num_sample, input_size, in_ch, iconfig, econfig):
        super(SetAgregate, self).__init__()
        self.num_sample= num_sample
        self.channels=in_ch
        self.input_size= input_size
        self.intra_setpool = SetPool(**iconfig)
        self.inter_setpool = SetPool(**econfig)
        self.fc_out = nn.Linear(512, input_size*input_size)

    def forward(self, X):
        proto_batch = []
        for x in X:
            x =x.cuda()

            cls_protos = self.intra_setpool(x).squeeze(1)
            proto_batch.append(self.inter_setpool(cls_protos.unsqueeze(0)))
        v = torch.stack(proto_batch).squeeze()
        v = self.fc_out(v)
        v = v.reshape((-1, self.channels, self.input_size, self.input_size))
        return v



#
#
# class SetAgregate(nn.Module):
#     def __init__(self, num_sample, iconfig, econfig):
#         super(SetAgregate, self).__init__()
#         self.num_sample= num_sample
#         self.intra_setpool = SetPool(**iconfig)
#         self.inter_setpool = SetPool(**econfig)
#         self.fc_out = nn.Linear(512, 1024)
#
#     def forward(self, X):
#         proto_batch = []
#         # print(len(X))
#         for x in X:
#             if isinstance(x[0], list):
#                 y = torch.stack(x[0], 0)
#             else:
#                 y = torch.stack(x, 0)
#             y = y.cuda()
#             # x = torch.cat(x, 0)
#             # y = y.cuda()
#
#             cls_protos = self.intra_setpool(y).squeeze(1)
#             proto_batch.append(self.inter_setpool(cls_protos.unsqueeze(0)))
#         v = torch.stack(proto_batch).squeeze()
#         v = self.fc_out(v)
#         v = v.reshape((-1, 1, 32, 32))
#         # print(v.shape)
#         # print('===============================')
#         return v




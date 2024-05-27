import torch
from torch import nn
from stage2.set_transformer.setencode import SetPool
from stage2.set_transformer.models import SetTransformer

class SetAgregate(nn.Module):
    def __init__(self, intraconfig, interconfig, in_ch=1, input_size=32, nz =56, ckpt_path=None, keep_keys=[], **kwargs):
        super(SetAgregate, self).__init__()
        self.input_size=input_size
        self.in_ch=in_ch
        self.ckpt_path = ckpt_path
        self.intra_setpool = SetPool(**intraconfig)
        self.inter_setpool = SetPool(**interconfig)
        self.set_proj= nn.Sequential(nn.Linear(nz, in_ch*input_size*input_size))

        if ckpt_path is not None:
            keep_keys = list(self.state_dict())
            self.init_from_ckpt(ckpt_path, keep_keys=keep_keys)
            for param in self.intra_setpool.parameters():
                param.requires_grad = False
            for param in self.inter_setpool.parameters():
                param.requires_grad = False

    def init_from_ckpt(self, path, keep_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            if k not in keep_keys:
                del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, X):
        proto_batch = []
        # print(X.shape)
        for x in X:
            if isinstance(x, list):
                # assert  len(x)==1, ' inner loop input  should be 1'
                x = x[0]
            x =x.cuda()
            # print(x.shape)
            cls_protos = self.intra_setpool(x).squeeze(1)
            proto_batch.append(self.inter_setpool(cls_protos.unsqueeze(0)))
        v = torch.stack(proto_batch).squeeze()
        # print(v.shape)
        out = self.set_proj(v)
        out = out.reshape(-1, self.in_ch, self.input_size, self.input_size)
        # print(out.shape)

        return out



class EmbedData(nn.Module):
    def __init__(self,  enconfig, deconfig, emb_dim=1024,in_ch=1, input_size=32, ckpt_path=None,  **kwargs):
        super(EmbedData, self).__init__()
        self.input_size = input_size
        self.in_ch = in_ch
        self.ckpt_path = ckpt_path

        # self.set_proj = nn.Sequential(nn.Linear(nz, in_ch * input_size * input_size))

        self.intra = SetTransformer(enconfig, deconfig)
        self.inter = SetTransformer(enconfig, deconfig)
        self.emb_dim = emb_dim
        # self.input_size= input_size
        # self.channels = channels
        self.proj = nn.Linear(512, emb_dim)

        if ckpt_path is not None:
            keep_keys = list(self.state_dict())
            self.init_from_ckpt(ckpt_path, keep_keys=keep_keys)
            for param in self.intra.parameters():
                param.requires_grad = False
            for param in self.inter.parameters():
                param.requires_grad = False

    def init_from_ckpt(self, path, keep_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            if k not in keep_keys:
                del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, inputs):

        outputs = []
        # print(len(inputs))
        for x in inputs:
            if isinstance(x, list) and len(x)==1:
                x= x[0]

            x =x.cuda()
            z = self.intra(x).squeeze(1)
            z = z.unsqueeze(0)
            out = self.inter(z).reshape(-1)
            outputs.append(out)
        outputs = torch.stack(outputs, 0).reshape(-1, 512)
        outputs = self.proj(outputs)
        outputs = outputs.reshape(-1, self.in_ch, self.input_size, self.input_size)
        return outputs


from stage2.set_transformer.modules import *
# from modules.settmodules import *
# from utils import instantiate_from_config
class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X

class encoder(nn.Module):
    def __init__(self, dim_input,  num_inds=32, dim_hidden=128, num_heads=4, ln=False,  **ignore_kwargs):
        super(encoder, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))

    def forward(self, x):
        return self.enc(x)


class decoder(nn.Module):
    def __init__(self, num_outputs, dim_output, dim_hidden=128, num_heads=4, ln=False, **ignore_kwargs):
        super(decoder, self).__init__()
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output))

    def forward(self, x):
        return self.dec(x)


class SetTransformer(nn.Module):
    def __init__(self, enconfig, deconfig):
        super(SetTransformer, self).__init__()
        # enconfig=config.params.encoder
        # deconfig=config.params.decoder
        self.enc = encoder(**enconfig)
        self.dec = decoder(**deconfig)


    def forward(self, X):
        x =self.enc(X)
        # print(x.shape)
        out = self.dec(x)
        return out
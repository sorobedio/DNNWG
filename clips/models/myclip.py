import torch
from torch import nn
import torch.nn.functional as F
from utils.util import instantiate_from_config

from clips.modules.modules import ProjectionHead, WeightEncoder


class CLIPModel(nn.Module):
    def __init__(self,
                 weight_encoder_config,
                 dataset_encoder_config,
                 temperature,
                 d_embedding,
                 w_embedding,
                 projection_dim,
                 dropout=0.0,
                 cond_key='dataset',
                 input_key='weight',
                 ckpt_path=None,
                 dataset_encoder_trainable=True,
                 weight_encoder_trainable=False,
                 device="cuda",
                 ignore_keys=[],
                 *args, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.d_embedding = d_embedding
        self.w_embedding = w_embedding
        self.weight_encoder_trainable = weight_encoder_trainable
        self.projection_dim = projection_dim
        self.dropout = dropout
        self.device = device
        self.input_key = input_key
        self.cond_key = cond_key
        self.dataset_encoder_trainable=dataset_encoder_trainable

        self.instantiate_dataset_encoder(dataset_encoder_config)
        self.instantiate_weight_encoder(weight_encoder_config)

        # self.dataset_projection = ProjectionHead(embedding_dim=d_embedding,
        #                                          projection_dim=projection_dim,
        #                                          dropout=dropout)
        # self.weight_projection = ProjectionHead(embedding_dim=w_embedding,
        #                                          projection_dim=projection_dim,
        #                                          dropout=dropout)
        self.temperature = temperature
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def instantiate_dataset_encoder(self, config):
        model = instantiate_from_config(config)
        for param in model.parameters():
            param.requires_grad = self.dataset_encoder_trainable
        self.dataset_encoder = model
        self.dataset_encoder = self.dataset_encoder.to(self.device)


    def instantiate_weight_encoder(self, config):
        model = instantiate_from_config(config)
        self.weight_encoder = model
        for param in self.weight_encoder.parameters():
            param.requires_grad = self.weight_encoder_trainable
        self.weight_encoder = self.weight_encoder.to(self.device)

    def forward(self, batch):
        # Getting dataset and weights Features
        y = batch["dataset"]
        w = batch["weight"].to(self.device)

        # _, prior = self.weight_encoder(batch["weight"].to(self.device))
        # weight_features = prior.sample()
        # weight_features = self.weight_encoder(y)
        # weight_features = torch.flatten(weight_features, start_dim=1)
        dataset_embeddings = self.dataset_encoder(y)
        weight_embeddings = self.weight_encoder(w)
        weight_embeddings = torch.flatten(weight_embeddings, start_dim=1)

        # dataset_embeddings = self.dataset_projection(dataset_features)
        # weight_embeddings = self.weight_projection(weight_features)

        # Calculating the Loss
        logits = (weight_embeddings @ dataset_embeddings.T) / self.temperature
        dataset_similarity = dataset_embeddings @ dataset_embeddings.T
        weight_similarity = weight_embeddings @ weight_embeddings.T
        targets = F.softmax(
            (dataset_similarity + weight_similarity) / 2 * self.temperature, dim=-1
        )
        weight_loss = cross_entropy(logits, targets, reduction='none')
        dataset_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (dataset_loss + weight_loss) / 2.0 #+ F.mse_loss(dataset_embeddings, weight_embeddings)# shape: (batch_size)
        return loss.mean()





class myCLIPModel(nn.Module):
    def __init__(self,
                 first_stage_config,
                 dataset_encoder_config,
                 temperature,
                 d_embedding,
                 w_embedding,
                 projection_dim,
                 dropout=0.0,
                 cond_key='dataset',
                 input_key='weight',
                 ckpt_path=None,
                 dataset_encoder_trainable=True,
                 device="cuda",
                 ignore_keys=[],
                 *args, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.d_embedding = d_embedding
        self.w_embedding = w_embedding
        self.projection_dim = projection_dim
        self.dropout = dropout
        self.device = device
        self.input_key = input_key
        self.cond_key = cond_key
        self.dataset_encoder_trainable=dataset_encoder_trainable

        self.instantiate_dataset_encoder(dataset_encoder_config)
        self.instantiate_weight_encoder(first_stage_config)

        self.dataset_projection = ProjectionHead(embedding_dim=d_embedding,
                                                 projection_dim=projection_dim,
                                                 dropout=dropout)
        self.weight_projection = ProjectionHead(embedding_dim=w_embedding,
                                                 projection_dim=projection_dim,
                                                 dropout=dropout)
        self.temperature = temperature
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def instantiate_dataset_encoder(self, config):
        model = instantiate_from_config(config)
        for param in model.parameters():
            param.requires_grad = self.dataset_encoder_trainable
        self.dataset_encoder = model
        self.dataset_encoder = self.dataset_encoder.to(self.device)


    def instantiate_weight_encoder(self, config):
        model = instantiate_from_config(config)
        self.weight_encoder = model
        self.weight_encoder = self.weight_encoder.to(self.device)

    def forward(self, batch):
        # Getting dataset and weights Features
        dataset_features = self.dataset_encoder(batch["dataset"].to(self.device))
        # _, prior = self.weight_encoder(batch["weight"].to(self.device))
        # weight_features = prior.sample()
        weight_features = self.weight_encoder(batch["weight"].to(self.device))
        weight_features= weight_features.reshape(-1, self.w_embedding)

        dataset_embeddings = self.dataset_projection(dataset_features)
        weight_embeddings = self.weight_projection(weight_features)

        # Calculating the Loss
        logits = (weight_embeddings @ dataset_embeddings.T) / self.temperature
        dataset_similarity = dataset_embeddings @ dataset_embeddings.T
        weight_similarity = weight_embeddings @ weight_embeddings.T
        targets = F.softmax(
            (dataset_similarity + weight_similarity) / 2 * self.temperature, dim=-1
        )
        weight_loss = cross_entropy(logits, targets, reduction='none')
        dataset_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (dataset_loss + weight_loss) / 2.0 # shape: (batch_size)
        return loss.mean()





def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

# if __name__ == '__main__':
#     images = torch.randn(8, 3, 224, 224)
#     input_ids = torch.randint(5, 300, size=(8, 25))
#     attention_mask = torch.ones(8, 25)
#     batch = {
#         'image': images,
#         'input_ids': input_ids,
#         'attention_mask': attention_mask
#     }
#
#     CLIP = CLIPModel()
#     loss = CLIP(batch)
#     print("")
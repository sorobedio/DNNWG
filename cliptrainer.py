import argparse, os, sys, datetime, glob
import numpy as np
import time
import itertools
import torch
import torchvision
import pytorch_lightning as pl
from torch.linalg import multi_dot
from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset
from functools import partial
from PIL import Image
from helpers.helpers import *
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from zoodatasets.sampler import ZooDataset

# from data.base import Txt2ImgIterableBaseDataset
from utils.util import instantiate_from_config
# from utils import AvgMeter, get_lr
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(description='clip encoder Training')
    parser.add_argument('--data', default='dataset', type=str, help='dataset root')
    parser.add_argument('--data_root', default='../Datasets/', type=str, help='dataset root for cifar10, mnist, ..')
    parser.add_argument('--topk', default=None, type=int, help='number of sample per dataset in training loader')
    parser.add_argument('--dataset', default='joint', type=str, help='dataset choice amoung'
                                                                     ' [mnist, svhn, cifar10, stl10, joint')
    parser.add_argument('--split', default='train', type=str, help='dataset split{ train, test, val]')
    parser.add_argument('--ae_type', default='ldm', type=str, help='auto encoder type [ldm, vqvae, simple]')
    parser.add_argument('--save_path', default='clipcheckpoints', type=str, help='checkpointys folders')
    parser.add_argument('--gpus', default=0, type=int, help='device')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decaay')
    parser.add_argument('--patience', default=10, type=int, help='scheduler patience')
    parser.add_argument('--factor', default=0.5, type=float, help='scheduler param')
    # parser.add_argument('--num_workers', default=4, type=int, help='device')

    parser.add_argument('--n_epochs', default=100, type=int, help='max epoch')
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="adt",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default='clips/configs/base_config.yaml',

    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))




class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        # batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["weight"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        # batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["weight"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():


    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.patience, factor=args.factor
    )
    step = "epoch"

    best_loss = float('inf')
    best_train_loss = float('inf')
    for epoch in range(args.n_epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "save best val checkpoint")
            print("Saved Best Model!")

        if train_loss.avg  < best_train_loss:
            best_train_loss = train_loss.avg
            torch.save(model, "clipcheckpoints/save full model")
            print("Saved Best training Model!")
        print(f'best train loss is : {best_train_loss}')



def my_collate(batch):
    sample = {}
    data = [item['weight'] for item in batch]
    conds = [item['dataset'] for item in batch]
    target =[]
    for items in conds:
        target += [item for item in items]
    data = torch.cat(data, 0)
    sample['weight']=data
    sample['dataset'] = target

    return sample


def m_collate(batch):
    sample = {}

    data = [item['weight'] for item in batch]
    cond = [item['dataset'] for item in batch]
    data = torch.stack(data, 0).type(torch.float32)
    sample['weight']=data
    sample['dataset'] = cond

    return sample


if __name__ == "__main__":


    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()

    args = parser.parse_args()
    opt, unknown = parser.parse_known_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainset = ZooDataset(root=args.data, dataset=args.dataset, split=args.split, normalize=False, num_sample=5)
    valset = ZooDataset(root=args.data, dataset=args.dataset, split='val', normalize=False, num_sample=5)

    # train_loader = DataLoader(trainset, shuffle=True, batch_size=100, num_workers=4)
    # valid_loader = DataLoader(valset, shuffle=False, batch_size=100, num_workers=4)
    train_loader = DataLoader(trainset, shuffle=True, batch_size=64, collate_fn=my_collate, num_workers=4)
    valid_loader = DataLoader(valset, shuffle=False, batch_size=24, collate_fn=my_collate)
    nowname= opt.name+now
    print(opt.base)
    print('----------------------')
    configs = [OmegaConf.load(opt.base)]

    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    model = instantiate_from_config(config.model)
    # args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    main()


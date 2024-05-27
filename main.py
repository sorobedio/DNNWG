import argparse, os, sys, datetime, glob
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

# from data.base import Txt2ImgIterableBaseDataset
from utils.util import instantiate_from_config

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

    parser = argparse.ArgumentParser(description='Autoencoder Training')
    parser.add_argument('--data_dir', default='../Datasets/minizoo', type=str, help='dataset root')
    parser.add_argument('--data_root', default='../Datasets/', type=str, help='dataset root for cifar10, mnist, ..')
    parser.add_argument('--topk', default=30, type=int, help='number of sample per dataset in training loader')
    parser.add_argument('--dataset', default='joint', type=str, help='dataset choice amoung'
                                                                     ' [mnist, svhn, cifar10, stl10, joint')
    parser.add_argument('--split', default='train', type=str, help='dataset split{ train, test, val]')
    parser.add_argument('--ae_type', default='ldm', type=str, help='auto encoder type [ldm, vqvae, simple]')
    parser.add_argument('--save_path', default='ae_checkpoints', type=str, help='checkpointys folders')
    parser.add_argument('--gpus', default=4, type=int, help='device')
    # parser.add_argument('--num_workers', default=4, type=int, help='device')

    parser.add_argument('--n_epochs', default=1000000, type=int, help='max epoch')
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
        help="paths to base configs. Loaded from left-to-right. base_config_kl.yaml"
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",

        default="stage1/configs/base_config_kl.yaml",

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




if __name__ == "__main__":


    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    # parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()
    nowname= opt.name+now
    configs = [OmegaConf.load(opt.base)]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    model = instantiate_from_config(config.model)


    ds = instantiate_from_config(config.data)
    ds.prepare_data()
    ds.setup(stage='fit')

    print("#### Data #####")
    print(f'dataset {ds.dataset}')
    # trainer = pl.Trainer( accumulate_grad_batches=4, accelerator="gpu", devices=1, min_epochs=10000,
    #                       max_epochs=100000)
    checkpoint_callback = ModelCheckpoint(monitor='train/aeloss',
                                          dirpath='vae_checkpoints/',
                                          filename='checkpoint_vae_model_{epoch}_',
                                          every_n_epochs=1
                                          )

    trainer = pl.Trainer(accelerator="gpu", devices=-1, min_epochs=100,
                         max_epochs=3000, log_every_n_steps=1, callbacks=[checkpoint_callback])
    trainer.fit(model, ds)

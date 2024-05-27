from helpers.helpers import *
from loaders.testloaders import get_loader
from helpers.misc import progress_bar

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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    parser.add_argument('--data', default='../Datasets/Tiny_imagenet/', type=str, help='dataset root')
    parser.add_argument('--data_root', default='../Datasets/', type=str, help='dataset root for cifar10, mnist, ..')
    parser.add_argument('--topk', default=30, type=int, help='number of sample per dataset in training loader')
    parser.add_argument('--dataset', default='joint', type=str, help='dataset choice amoung'
                                                                     ' [mnist, svhn, cifar10, stl10, joint')
    parser.add_argument('--split', default='train', type=str, help='dataset split{ train, test, val]')
    parser.add_argument('--ae_type', default='ldm', type=str, help='auto encoder type [ldm, vqvae, simple]')
    parser.add_argument('--save_path', default='aecheckpoints', type=str, help='checkpointys folders')
    parser.add_argument('--gpus', default=0, type=int, help='device')
    parser.add_argument('--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision(16-bit) ')
    # parser.add_argument('--num_workers', default=4, type=int, help='device')

    # parser.add_argument('--n_epochs', default=1000000, type=int, help='max epoch')
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

        default="stage2/configs/base_config.yaml",

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


from glob import glob
def validate(val_loader, model, criterions):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterions(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def get_dataset_condition(xc, n= 1, num_sample=5):
    # xc = torch.load(fp)
    keys = list(xc)
    conds = {}
    # num_class = 10
    for k in keys:
        x = xc[k]
        num_class = len(x)
        data = []
        classes = list(range(num_class))
        xd = []
        for i in range(n):
            cdata = []
            for cls in classes:
                # cx = x[cls][0]
                cx = x[cls]
                ridx = torch.randperm(len(cx))
                cdata.append(cx[ridx][:num_sample])
            xd.append(torch.stack(cdata, 0))
        y = torch.stack(xd, 0).type(torch.float32)
        # print(y.shape)
        conds[k] =y
    return conds



def sample_cond(x, n=3, num_sample=5):
    # xc = torch.load(fp)
    # x = xc[dset]
    num_class = len(x)
    classes = list(range(num_class))
    xd = []
    for i in range(n):
        cdata = []
        for cls in classes:
            # cx = x[cls][0]
            cx = x[cls]
            ridx = torch.randperm(len(cx))
            cdata.append(cx[ridx][:num_sample])
        xd.append(torch.stack(cdata, 0))
    x = torch.stack(xd, 0)

    return x

def split_sample_cond(x, split=100, n=3, num_sample=5):
    # xc = torch.load(fp)
    # x = xc[dset]
    data = {}
    num_class = len(x)


    classes = list(range(num_class))
    xd = []
    for i in range(n):
        cdata = []
        for cls in classes:
            # cx = x[cls][0]
            cx = x[cls]
            ridx = torch.randperm(len(cx))
            cdata.append(cx[ridx][:num_sample])
        xd.append(torch.stack(cdata, 0))
    x = torch.stack(xd, 0)

    return x


from zoomodels.clizoomodel import *
# from data_utils.kaggle_loader import load_data
# from data_utils.tinyloader import load_data
from data_utils.imnetloader import load_base_data
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics

if __name__ == '__main__':
    # define loss function (criterion) and optimizer
    criterions = nn.CrossEntropyLoss().cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # sys.path.append(os.getcwd())

    parser = get_parser()
    args = parser.parse_args()
    opt, unknown = parser.parse_known_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nowname = opt.name + now
    print(opt.base)
    print('----------------------')
    configs = [OmegaConf.load(opt.base)]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    # autoencoder = instantiate_from_config(config.model)
    # args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ldmmodel = instantiate_from_config(config.model)

    std = torch.load('ldm_checkpoints/checkpoint_ldm_model.pt')['state_dict']
    ldmmodel.load_state_dict(std)
    ldmmodel = ldmmodel.to(device)
    ldmmodel.eval()


    root = 'path to dataset'
    conds = torch.load('path to conditioned features dataset')

    ##################################################333

    dsets = list(conds)


    scale = 1.0
    split = False
    for dset in dsets:
        val_loader, val_loaders, n_classes = load_base_data(root, dset, batch_size=512)
        print(f'==> evating on {dset}......{n_classes}........')
        x = conds[dset]
        xc = sample_cond(x, n=50, num_sample=5)
        print(xc.shape)
        xc = xc.type(torch.float32)
        wl =[]
        pw = []
        pb =[]
        #
        weight = ldmmodel.sample(cond=xc.to(device))
        ac =[]
        best =0.0
        for w in weight:
            pass
        #     model = Classifier(in_dim=768, n_classes=n_classes)
        #     set_weights(model, w * scale)
        #     model.cuda()
        #     acc = validate(val_loader, model, criterions)
        #     print(f'test accuracy===== {acc} =========')
        #     if acc > best:
        #         best = acc
        #     ac.append(acc)
        # ac = sorted(ac, reverse=True)
        # print(best)
        # print(f' {dset} mean: {np.mean(ac[:3])} std: {np.std(ac[:3])}')

        print('**********************************************************')


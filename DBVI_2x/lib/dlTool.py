import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from configs.configTrain import configMain
import math
import torch
from sklearn.mixture import GaussianMixture
import numpy as np
from torch.optim import Adam


def getOptimizer(gNet: nn.Module, cfg: configMain):
    core_network_params = []
    map_network_params = []
    for k, v in gNet.named_parameters():
        if v.requires_grad:
            if 'mapping' in k:
                map_network_params.append(v)
            else:
                core_network_params.append(v)
    optim = Adam([{'params': core_network_params, 'lr': cfg.optim.lrInit},
                  {'params': map_network_params, 'lr': 1e-2 * cfg.optim.lrInit}],
                 weight_decay=0, betas=(0.9, 0.999))
    for group in optim.param_groups:
        group.setdefault('initial_lr', group['lr'])
    maxPSNR = -1
    return optim, maxPSNR


def getScheduler(optimizer, epoch=-1):
    return lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[110, 135], gamma=0.4, last_epoch=epoch)


def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(
        kernel_size
    ), "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


def fitFullConvGaussian(zAll: torch.Tensor):
    zAllNpy = zAll.detach().cpu().numpy()
    selectIdx = np.random.choice(zAllNpy.shape[0], size=100, replace=False)
    zAllNpySellect = zAllNpy[selectIdx, ...]
    gmm = GaussianMixture(n_components=10, covariance_type='full').fit(zAllNpySellect)
    return gmm

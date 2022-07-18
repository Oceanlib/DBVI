import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from lib.softsplat import FunctionSoftsplat


def backWarp(img: torch.Tensor, flow: torch.Tensor):
    device = img.device
    N, C, H, W = img.size()

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    gridY, gridX = torch.meshgrid([torch.arange(start=0, end=H, device=device, requires_grad=False),
                                   torch.arange(start=0, end=W, device=device, requires_grad=False)])

    x = gridX.unsqueeze(0).expand_as(u).float().detach() + u
    y = gridY.unsqueeze(0).expand_as(v).float().detach() + v

    # range -1 to 1
    x = 2 * x / (W - 1.0) - 1.0
    y = 2 * y / (H - 1.0) - 1.0
    # stacking X and Y
    grid = torch.stack((x, y), dim=3)

    imgOut = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    mask = torch.ones_like(img, requires_grad=False)

    mask = F.grid_sample(mask, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return imgOut * (mask.detach()), mask.detach()
    # return imgOut


class ModuleSoftsplat(torch.nn.Module):
    def __init__(self, strType='average'):
        super().__init__()

        self.strType = strType

    def forward(self, tenInput, tenFlow, tenMetric):
        return FunctionSoftsplat(tenInput, tenFlow, tenMetric, self.strType)


class fidelityGradCuda(nn.Module):
    def __init__(self, res=True):
        super(fidelityGradCuda, self).__init__()
        self.fWarp = ModuleSoftsplat()

    def forward(self, It: torch.Tensor, I0: torch.Tensor, F0t: torch.Tensor):
        self.device = I0.device
        It0, mask = backWarp(It, F0t)
        grad_ll0 = (I0 - It0)  # grad(y=0.5*(I0 - It0)^2, x=It0)

        totalGrad = grad_ll0 * mask
        warpGrad = self.fWarp(tenInput=totalGrad, tenFlow=F0t, tenMetric=None)

        return warpGrad


class fidelityGradTorch(nn.Module):
    def __init__(self):
        super(fidelityGradTorch, self).__init__()

    def forward(self, It: torch.Tensor, I0: torch.Tensor, F0t: torch.Tensor):
        with torch.enable_grad():
            It.requires_grad_()
            It0, mask = backWarp(It, F0t)
            loss = -(0.5 * (I0 * mask - It0 * mask) ** 2).sum()
            warpGrad = grad(loss, It, create_graph=True)[0]

        return warpGrad
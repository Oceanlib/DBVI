import torch.nn.functional as F
import torch
from torch import nn


class l1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(l1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, gt, mask=None):
        if mask is None:
            return F.l1_loss(pred, gt, reduction=self.reduction)
        else:
            return F.l1_loss(pred * mask, gt * mask, reduction=self.reduction)

    def __repr__(self):
        return 'l1Loss'


class l2Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(l2Loss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, gt, mask=None):
        if mask is None:
            return F.mse_loss(pred, gt, reduction=self.reduction)
        else:
            return F.mse_loss(pred * mask, gt * mask, reduction=self.reduction)

    def __repr__(self):
        return 'l2Loss'


class totalLoss(nn.Module):
    def __init__(self):
        super(totalLoss, self).__init__()
        self.l1loss = l1Loss()
        self.scale = [1, 1, 1, 1, 1, 1]
        self.lossNames = ['l1']
        self.lossDict = {}

    def forward(self, rgbs, gts):
        self.lossDict.clear()
        l1loss = 0
        if len(rgbs) > 1:
            assert all([len(self.scale) == len(rgbs), len(rgbs) == len(gts)])
            scales = self.scale
        else:
            scales = self.scale[-len(rgbs)::]
        for scale, rgb, gt in zip(scales, rgbs, gts):
            l1loss += self.l1loss(rgb, gt.detach()).mean() * scale * 1

        lossSum = l1loss

        self.lossDict.setdefault('l1', l1loss.detach())
        self.lossDict.setdefault('Total', lossSum.detach())

        return lossSum


if __name__ == '__main__':
    a = torch.randn([1, 3, 128, 128])
    b = torch.randn([1, 3, 128, 128])
    mask = torch.ones_like(a)

    loss = totalLoss()

    lossOut = loss(a, b, mask)
    lossOut2 = loss(a, b, mask)
    pass

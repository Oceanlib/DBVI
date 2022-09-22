import torch
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
import torch.nn.functional as F
from math import exp
from torch import nn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


class skiMetric(object):
    def __init__(self, drange=1.0):
        super(skiMetric, self).__init__()
        self.range = drange

    def psnr(self, img: np.ndarray, gt: np.ndarray):
        """
        :param img:H, W, C
        :param gt: H, W, C
        :return:
        """
        assert img.shape == gt.shape, 'img shape != gt shape'
        assert 3 == img.ndim
        return compare_psnr(img, gt, data_range=self.range)

    def ssim(self, img: np.ndarray, gt: np.ndarray):
        """
        :param img:H, W, C
        :param gt: H, W, C
        :return:
        """
        assert img.shape == gt.shape, 'img shape != gt shape'
        assert 3 == img.ndim
        return compare_ssim(img, gt, data_range=self.range, multichannel=True, gaussian_weights=True)


class torchMetricPSNR():
    def __init__(self, drange=None, reduce='mean'):
        super(torchMetricPSNR, self).__init__()
        self.range = drange
        self.reduce = reduce
        self.window_size = 11
        self.device0 = None

    def getRange(self, img):
        if torch.max(img) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img) < -0.5:
            min_val = -1
        else:
            min_val = 0

        return max_val - min_val

    def psnr(self, img: torch.Tensor, gt: torch.Tensor):
        # self.device0 = img.device
        # gt = gt.to(self.device0).detach()
        assert all([4 == img.ndimension(), 4 == gt.ndimension(), img.shape == gt.shape])

        self.range = self.getRange(gt)

        N, C, H, W = img.shape

        mse = ((img - gt.detach()) ** 2).mean(dim=[1, 2, 3])  # N, 1
        psnrBatch = 10 * torch.log10(self.range ** 2 * mse.reciprocal())  # N, 1

        if self.reduce == 'sum':
            psnrBatch = psnrBatch.sum()
        elif self.reduce == 'mean':
            psnrBatch = psnrBatch.mean()

        return psnrBatch, N

    def ssim(self, img, gt):
        self.device0 = img.device
        self.range = self.getRange(img)

        padd = 0
        N, C, H, W = img.size()

        real_size = min(self.window_size, H, W)
        sigma = 1.5
        gaussList = [exp(-(x - real_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(real_size)]
        gauss = torch.tensor(gaussList)
        _1D_window = (gauss / gauss.sum()).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
        window = _3D_window.expand(1, 1, real_size, real_size, real_size).contiguous().to(self.device0)

        img = img.unsqueeze(1)
        gt = gt.unsqueeze(1)

        mu1 = F.conv3d(F.pad(img, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)
        mu2 = F.conv3d(F.pad(gt, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv3d(F.pad(img * img, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd,
                             groups=1) - mu1_sq
        sigma2_sq = F.conv3d(F.pad(gt * gt, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd,
                             groups=1) - mu2_sq
        sigma12 = F.conv3d(F.pad(img * gt, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd,
                           groups=1) - mu1_mu2

        C1 = (0.01 * self.range) ** 2
        C2 = (0.03 * self.range) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)

        ssim_map: torch.Tensor = (((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)).mean(dim=[1, 2, 3])

        if self.reduce == 'sum':
            ssim_map = ssim_map.sum()
        elif self.reduce == 'mean':
            ssim_map = ssim_map.mean()
        return ssim_map, N


class torchMetric(nn.Module):
    def __init__(self, reduce='sum', device=torch.device('cpu'), ifssim=True):
        super(torchMetric, self).__init__()
        self.range = 1.0
        self.reduce = reduce
        self.window_size = 11
        self.device = device
        if ifssim:
            self.register_buffer('window', self.initWindow())

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, ifssim=True):
        N, C, H, W = pred.shape
        pred = self.reRange(pred)
        gt = self.reRange(gt)
        psnr = self.psnr(pred.detach(), gt.detach())
        if ifssim:
            ssim = self.ssim(pred.detach(), gt.detach())
        else:
            ssim = psnr

        return psnr, ssim, N

    # @torch.jit.script
    def psnr(self, img: torch.Tensor, gt: torch.Tensor):
        assert all([4 == img.ndimension(), 4 == gt.ndimension(), img.shape == gt.shape])

        mse = ((img - gt.detach()) ** 2).mean(dim=[1, 2, 3])  # N, 1
        psnrBatch = 10 * torch.log10(self.range ** 2 * mse.reciprocal())  # N, 1

        if self.reduce == 'sum':
            psnrBatch = psnrBatch.sum(dim=0)
        elif self.reduce == 'mean':
            psnrBatch = psnrBatch.mean()

        return psnrBatch

    # @torch.jit.script
    def ssim(self, img: torch.Tensor, gt: torch.Tensor):
        padd = 0
        img = img.unsqueeze(1)
        gt = gt.unsqueeze(1)

        mu1 = F.conv3d(F.pad(img, (5, 5, 5, 5, 5, 5), mode='replicate'), self.window, padding=padd, groups=1)
        mu2 = F.conv3d(F.pad(gt, (5, 5, 5, 5, 5, 5), mode='replicate'), self.window, padding=padd, groups=1)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv3d(F.pad(img * img, (5, 5, 5, 5, 5, 5), 'replicate'), self.window, padding=padd,
                             groups=1) - mu1_sq
        sigma2_sq = F.conv3d(F.pad(gt * gt, (5, 5, 5, 5, 5, 5), 'replicate'), self.window, padding=padd,
                             groups=1) - mu2_sq
        sigma12 = F.conv3d(F.pad(img * gt, (5, 5, 5, 5, 5, 5), 'replicate'), self.window, padding=padd,
                           groups=1) - mu1_mu2

        # C1 = (0.01 * self.range) ** 2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)

        ssim_map: torch.Tensor = (((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)).mean(dim=[1, 2, 3, 4])

        if self.reduce == 'sum':
            ssim_map = ssim_map.sum(dim=0)
        elif self.reduce == 'mean':
            ssim_map = ssim_map.mean()
        return ssim_map

    def initWindow(self):
        wSize = self.window_size
        sigma = 1.5
        gaussList = [exp(-(x - wSize // 2) ** 2 / float(2 * sigma ** 2)) for x in range(wSize)]
        gauss = torch.tensor(gaussList)
        _1D_window = (gauss / gauss.sum()).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
        return _3D_window.expand(1, 1, wSize, wSize, wSize).contiguous()

    def reRange(self, img: torch.Tensor):
        if img.max() > 128:
            img = img.float() / 255.0
        if img.min() < -0.5:
            img = (img.float() + 1.0) / 2.0
        return img

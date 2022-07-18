from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F
from torch.nn import init
import numpy as np


class BaseNet(nn.Module):
    """
    init pretrained weights with weight file
    init other weights with a certain random pattern
    """

    def __init__(self):
        super(BaseNet, self).__init__()
        self.netInitType = 'kaiming'  # normal, xavier, orthogonal, kaiming
        self.netInitGain = 0.2

    def forward(self, *input):
        raise NotImplementedError

    def randomInitNet(self):
        # init all weights by pre-defined pattern firstly
        for m in self.modules():
            if any([isinstance(m, nn.Conv2d), isinstance(m, nn.ConvTranspose2d), isinstance(m, nn.Linear)]):
                if self.netInitType == 'normal':
                    init.normal_(m.weight, 0.0, std=self.netInitGain)
                elif self.netInitType == 'xavier':
                    init.xavier_normal_(m.weight, gain=self.netInitGain)
                elif self.netInitType == 'kaiming':
                    init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                elif self.netInitType == 'orthogonal':
                    init.orthogonal_(m.weight, gain=self.netInitGain)
                elif self.netInitType == 'default':
                    pass
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            elif any([isinstance(m, nn.InstanceNorm2d), isinstance(m, nn.LocalResponseNorm),
                      isinstance(m, nn.BatchNorm2d), isinstance(m, nn.GroupNorm)]):
                try:
                    init.constant_(m.weight, 1.0)
                    init.constant_(m.bias, 0.0)
                except Exception as e:
                    pass

    def initPreweight(self, pathPreWeight: str = None, rmModule=True):

        preW = self.getWeight(pathPreWeight)
        assert preW is not None, 'weighth in {} is empty'.format(pathPreWeight)
        modelW = self.state_dict()
        preWDict = OrderedDict()
        # modelWDict = OrderedDict()

        for k, v in preW.items():
            if rmModule:
                preWDict[k.replace('module.', "")] = v
            else:
                preWDict[k] = v

        shareW = {k: v for k, v in preWDict.items() if str(k) in modelW}
        assert shareW, 'shareW is empty'
        self.load_state_dict(preWDict, strict=False)

    @staticmethod
    def getWeight(pathPreWeight: str = None):
        if pathPreWeight is not None:
            return torch.load(pathPreWeight, map_location=torch.device('cpu'))
        else:
            return None

    @staticmethod
    def padToScale(img, netScale):
        _, _, h, w = img.size()
        oh = int(np.ceil(h * 1.0 / netScale) * netScale)
        ow = int(np.ceil(w * 1.0 / netScale) * netScale)
        img = F.pad(img, [0, ow - w, 0, oh - h], mode='reflect')
        return img


def normalize_tensor(in_feat: torch.Tensor):
    return F.normalize(in_feat, p=2, dim=1)


class conv(nn.Module):
    def __init__(self, inCh, outCh, kernel_size=3, sd=1, dilation=1, order='cnab', act='gelu', zeroInit=False, **args):
        super(conv, self).__init__()
        padding = self.get_valid_padding(kernel_size, dilation)
        bias = True if 'b' in order else False
        layers = []

        idxC = 0
        for idx, name in enumerate(order):
            if name == 'c':
                conv2d = nn.Conv2d(inCh, outCh, kernel_size, sd, padding=padding, dilation=dilation, bias=bias)
                if zeroInit:
                    conv2d.weight.data.fill_(0.0)
                    if bias:
                        conv2d.bias.data.fill_(0.0)
                layers.append(conv2d)
                idxC = idx
            if name == 'n':
                nch = outCh if idx > idxC else inCh
                layers.append(nn.GroupNorm(num_groups=4, num_channels=nch, affine=True))
            if name == 'a':
                if act == 'relu':
                    layers.append(nn.LeakyReLU(0.2, inplace=True))
                else:
                    layers.append(nn.GELU())  # GELU YYDS

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.block(x)

    @staticmethod
    def get_valid_padding(kernel_size, dilation):
        kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        padding = (kernel_size - 1) // 2
        return padding


# class upBlock(nn.Module):
#     def __init__(self, inCh, outCh, ks=5, sd=2, mode='nearest'):
#         super(upBlock, self).__init__()
#         self.upConv1 = nn.Sequential(nn.Upsample(scale_factor=sd, mode=mode),
#                                      conv(inCh, outCh, ks, norm=True, act=True, order='nac'))
#         self.downConv = conv(outCh, inCh, ks, sd=sd, norm=True, act=True, order='nac')
#
#         self.upConv2 = nn.Sequential(nn.Upsample(scale_factor=sd, mode=mode),
#                                      conv(inCh, outCh, ks, norm=True, act=True, order='nac'))
#
#     def forward(self, x):
#         h0 = self.upConv1(x)
#         l0 = self.downConv(h0)
#         h1 = self.upConv2(l0 - x)
#         return h1 + h0
#
#
# class downBlock(nn.Module):
#     def __init__(self, inCh, outCh, ks=5, sd=2, mode='nearest'):
#         super(downBlock, self).__init__()
#         self.downConv1 = conv(inCh, outCh, ks, sd=sd, norm=True, act=True, order='nac')
#         self.upConv = nn.Sequential(nn.Upsample(scale_factor=sd, mode=mode),
#                                     conv(outCh, inCh, ks, norm=True, act=True, order='nac'))
#         self.downConv2 = conv(inCh, outCh, ks, sd=sd, norm=True, act=True, order='nac')
#
#     def forward(self, x):
#         l0 = self.downConv1(x)
#         h0 = self.upConv(l0)
#         l1 = self.downConv2(h0 - x)
#         return l1 + l0
class PreActBottleneck(nn.Module):
    def __init__(self, inCh: int, midCh: int, outCh: int, sd: int = 1):

        super(PreActBottleneck, self).__init__()

        self.sd = sd
        self.conv0 = conv(inCh=inCh, outCh=midCh, kernel_size=1, order='nac')
        self.conv1 = conv(inCh=midCh, outCh=midCh, kernel_size=3, sd=sd, order='nac')
        self.conv2 = conv(inCh=midCh, outCh=outCh, kernel_size=1, order='nac')

        side = []
        if sd != 1:
            side.append(conv(inCh=inCh, outCh=inCh, kernel_size=3, sd=sd, order='nac'))
        if inCh != outCh:
            side.append(conv(inCh=inCh, outCh=outCh, kernel_size=1, sd=1, order='nac'))

        if side:
            self.side = nn.Sequential(*side)
        else:
            self.side = None

    def forward(self, x):
        residual = x

        out = self.conv2(self.conv1(self.conv0(x)))

        if self.side is not None:
            residual = self.side(x)

        out += residual

        return out


class ScalingLayer(nn.Module):
    # For rescaling the input to vgg16
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class RerangeLayer(nn.Module):
    # Change the input from range [-1., 1.] to [0., 1.]
    def __init__(self):
        super(RerangeLayer, self).__init__()

    def forward(self, inp):
        return (inp + 1.) / 2.


class NetLinLayer(nn.Module):
    ''' A single linear layer used as placeholder for LPIPS learnt weights '''

    def __init__(self, weight=None):
        super(NetLinLayer, self).__init__()
        self.register_buffer('weight', weight)

    def forward(self, inp: torch.Tensor):
        out = self.weight * inp
        return out


class downBlock(nn.Module):
    def __init__(self, inCh, outCh, ks=3, sd=2, order='nac'):
        super(downBlock, self).__init__()
        self.downConv1 = conv(inCh, outCh, kernel_size=ks, sd=sd, order=order)

    def forward(self, x):
        x1 = self.downConv1(x)
        # x2 = x1 + F.interpolate(x, scale_factor=0.5, mode='nearest', recompute_scale_factor=True)
        return x1


class upBlock(nn.Module):
    def __init__(self, inCh, outCh, ks=3, sd=2, mode='nearest'):
        super(upBlock, self).__init__()
        self.upConv1 = nn.Sequential(nn.Upsample(scale_factor=sd, mode='nearest'),
                                     conv(inCh, outCh, kernel_size=ks, order='nac'))

    def forward(self, x):
        x1 = self.upConv1(x)
        # x2 = x1 + F.interpolate(x, scale_factor=2, mode='nearest', recompute_scale_factor=True)
        return x1


# class upBlock(nn.Module):
#     def __init__(self, inCh, outCh, sf=2, ks=3, mode='nearest'):
#         super(upBlock, self).__init__()
#         self.up = nn.Upsample(scale_factor=sf, mode=mode)
#         self.conv = conv(inCh, outCh, ks, norm=True, act=True, order='nac')
#
#     def forward(self, x):
#         up = self.up(x)
#         return self.conv(up) + up

class RRUnet(nn.Module):
    def __init__(self, inCh, order='nac'):
        super(RRUnet, self).__init__()
        # conv in conv out
        midCh = [inCh, 16, 64, 256]

        # self.block1 = nn.Sequential(downBlock(inCh=midCh[0], outCh=midCh[1], norm=norm),
        #                             PreActBottleneck(inCh=midCh[1], midCh=midCh[1] // 2, outCh=midCh[1]))
        self.block1 = nn.Sequential(downBlock(inCh=midCh[0], outCh=midCh[1], order=order))

        # self.block2 = nn.Sequential(downBlock(inCh=midCh[1], outCh=midCh[2]),
        #                             PreActBottleneck(inCh=midCh[2], midCh=midCh[2] // 2, outCh=midCh[2]))

        self.block2 = nn.Sequential(downBlock(inCh=midCh[1], outCh=midCh[2]))

        self.block3 = nn.Sequential(downBlock(inCh=midCh[2], outCh=midCh[3]),
                                    PreActBottleneck(inCh=midCh[3], midCh=midCh[3] // 4, outCh=midCh[3]),
                                    PreActBottleneck(inCh=midCh[3], midCh=midCh[3] // 4, outCh=midCh[3])
                                    )

        # self.deBlock1 = nn.Sequential(upBlock(inCh=midCh[2], outCh=midCh[2]),
        #                               PreActBottleneck(inCh=midCh[2], midCh=midCh[2] // 2, outCh=midCh[1]))

        self.deBlock0 = nn.Sequential(upBlock(inCh=midCh[3], outCh=midCh[2]))

        self.deBlock1 = nn.Sequential(upBlock(inCh=midCh[2], outCh=midCh[1]))

        # self.deBlock2 = nn.Sequential(upBlock(inCh=midCh[1], outCh=midCh[1]),
        #                               PreActBottleneck(inCh=midCh[1], midCh=midCh[1] // 2, outCh=midCh[0]))
        self.deBlock2 = nn.Sequential(upBlock(inCh=midCh[1], outCh=midCh[0]))

    def forward(self, x):
        # x0 = self.block0(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        x4 = self.deBlock0(x3) + x2
        x5 = self.deBlock1(x4) + x1
        x6 = self.deBlock2(x5) + x

        return x6


class RRRB(nn.Module):
    def __init__(self, inCh, ks=3, order='nac'):
        super(RRRB, self).__init__()
        self.RDB1 = RRUnet(inCh, order=order)
        self.RDB2 = RRUnet(inCh)
        # self.RDB3 = RRUnet(inCh)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        # out = self.RDB3(out)
        return out + x


class StyleBlock(nn.Module):
    def __init__(self, ndBlock, outCh, midMap):
        super(StyleBlock, self).__init__()
        self.nb = ndBlock

        for i in range(ndBlock):
            if i == 0:
                self.add_module(f"{i:d}", RRRB(outCh, ks=3, order='nac'))
            else:
                self.add_module(f"{i:d}", RRRB(outCh, ks=3, order='ac'))
            self.add_module(f"trans{i:d}", nn.Linear(midMap, 2 * outCh))

        self.lrConv = conv(outCh, outCh, kernel_size=3, norm=False, act=True, order='nac')

    def forward(self, x, x_feat):
        bs, nc, w, h = x.shape
        shotCut = x

        for i in range(self.nb):
            rrdb_out = getattr(self, f"{i:d}")(x)
            tran_out = getattr(self, f"trans{i:d}")(x_feat)
            scale, bias = torch.chunk(tran_out, 2, dim=1)

            scale = torch.tanh(scale.view([bs, nc, 1, 1]))
            bias = bias.view([bs, nc, 1, 1])
            x = (1. + scale) * F.instance_norm(rrdb_out) + bias

        out = (self.lrConv(x) + shotCut) / 2.0
        return out
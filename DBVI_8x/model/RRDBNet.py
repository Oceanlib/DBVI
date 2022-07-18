import torch
from torch import nn
from model import block as B
from configs.configTrain import configMain
from lib.warp import fidelityGradCuda as fidelity
# from lib.warp import fidelityGradTorch as fidelity

from torch.nn import functional as F
from functools import partial
import math
from model.invBlock.permute import TransConv1x1


class subNet(B.BaseNet):
    def __init__(self, inCh=3, outCh=16, gradCh=8, gradFlowIn=4):
        super(subNet, self).__init__()

        midMap = 512
        mapping = [nn.Linear(128, midMap), nn.LeakyReLU(0.2, True)]
        for i in range(7):
            mapping.append(nn.Linear(midMap, midMap))
            mapping.append(nn.LeakyReLU(0.2, True))
        self.mapping = nn.Sequential(*mapping)

        self.convUV = nn.Sequential(B.conv(outCh, outCh, kernel_size=3, order='nac'),
                                    B.conv(outCh, gradCh * 2, kernel_size=1, order='nac', zeroInit=True))
        self.invConv = TransConv1x1(gradCh)

        self.compConv = B.conv(gradCh, gradFlowIn, kernel_size=1, order='nac')

        self.inConv = B.conv(inCh, outCh, kernel_size=3, norm=False, act=False, order='cna')

        self.style_block = B.StyleBlock(ndBlock=2, outCh=outCh, midMap=midMap)

        self.convNext = nn.Sequential(B.conv(outCh, outCh, kernel_size=3, norm=True, act=True, order='nac'),
                                      B.conv(outCh, outCh, kernel_size=3, norm=True, act=True, order='nac')
                                      )
        self.randomInitNet()

        self.toRGB = B.conv(outCh, 3, kernel_size=3, order='nac', zeroInit=True)

        self.toFlow = nn.Sequential(B.conv(outCh, outCh, kernel_size=3, order='nac'),
                                    B.conv(outCh, gradCh, kernel_size=1, order='nac', zeroInit=True))

    def forward(self, I0, I1, It, fid0, fid1, F0t, F1t, fea, d01):
        n, c, h, w = It.shape

        styleCode = torch.zeros([n, 128], dtype=torch.float32, device=It.device).detach()

        mu_invVar = self.convUV(fea)
        mu, invVar = torch.chunk(mu_invVar, chunks=2, dim=1)
        invVar = torch.sigmoid(invVar)

        d01Fea, _ = self.invConv(d01)
        gradFlow = self.negGradFlow(Fea=d01Fea, mean=mu, invVar=invVar)
        gradFlow = self.compConv(gradFlow)

        xCode = self.inConv(torch.cat([I0, I1, It, fid0, fid1, F0t, F1t, fea, gradFlow], dim=1))

        affineCode = self.mapping(styleCode)

        backbone = self.style_block(xCode, affineCode)

        feaNext = self.convNext(backbone)

        ItOut = self.toRGB(feaNext)  # [-1, 1]

        d01New = self.toFlow(feaNext)
        ItNew = It + ItOut

        return feaNext, ItNew, d01New

    def negGradFlow(self, Fea, mean=0.0, invVar=1.0):
        ngrad = (Fea - mean) * invVar
        gradFlow, _ = self.invConv(ngrad, rev=True)

        return gradFlow


class IMRRDBNet(B.BaseNet):
    def __init__(self, cfg: configMain):
        super(IMRRDBNet, self).__init__()
        self.netScale = 8

        I01Ch = 3 + 3
        ItCh = 3
        fidICh = 3 + 3  # grad(I0 ,WIt), grad(I0 ,WIt)
        flowCh = 2 + 2
        gradFlowCh = 16 + 16
        gradFlowIn = 4
        FeaCh = [16, 16, 16, 16, 16]  # Fea for next level

        self.scale = cfg.model.scale
        for idx, s in enumerate(self.scale):
            allInCh = I01Ch + ItCh + fidICh + flowCh + FeaCh[idx] + gradFlowIn
            allOutCh = FeaCh[idx + 1]

            self.add_module(f'fidelityGrad_{idx}', fidelity())
            self.add_module(f'subNet_{idx}', subNet(inCh=allInCh, outCh=allOutCh,
                                                    gradCh=gradFlowCh, gradFlowIn=gradFlowIn))

        if cfg.resume:
            self.initPreweight(cfg.path.ckpt)

    def forward(self, batchDict, flowNet, t: float = None):
        In, I0, I1, I2 = batchDict['In'], batchDict['I0'], batchDict['I1'], batchDict['I2']
        channelMean = sum([Ik.mean(dim=[2, 3], keepdim=True) for Ik in [In, I0, I1, I2]]) / 4.0
        In, I0, I1, I2 = [Ik - channelMean for Ik in [In, I0, I1, I2]]

        IB, IC, IH, IW = In.shape
        In, I0, I1, I2 = [self.padToScale(I, self.netScale) for I in [In, I0, I1, I2]]

        if t is None:
            t = batchDict['t'].view([In.shape[0], 1, 1, 1])
        else:
            t = torch.tensor(data=t, device=In.device, dtype=torch.float32).view([In.shape[0], 1, 1, 1])

        N, C, H, W = I0.shape

        F0t, F1t = self.getFt(In, I0, I1, I2, t, flowNet)

        output = []
        level = len(self.scale)

        fea = torch.zeros([N, 16, H, W], dtype=torch.float32, device=I0.device).detach()
        It = torch.zeros([N, C, H, W], dtype=torch.float32, device=I0.device).detach()
        df = torch.zeros([N, 32, H, W], dtype=torch.float32, device=I0.device).detach()

        for idx, l in enumerate(range(level)):

            net = getattr(self, f'subNet_{idx}')
            getFidGrad = getattr(self, f'fidelityGrad_{idx}')

            fid0 = getFidGrad(It=It, I0=I0, F0t=F0t)
            fid1 = getFidGrad(It=It, I0=I1, F0t=F1t)

            fea, It, d01t = net(I0, I1, It, fid0 * (1.0 - t), fid1 * t, F0t, F1t, fea, df)
            df = df + d01t

            d0t, d1t = torch.chunk(d01t[:, 0:4, :, :], chunks=2, dim=1)
            F0t = F0t + d0t
            F1t = F1t + d1t
            output.append(It[:, :, 0:IH, 0:IW] + channelMean)

        return output

    def adap2Net(self, tensor: torch.Tensor):
        Height, Width = tensor.size(2), tensor.size(3)

        Height_ = int(math.floor(math.ceil(Height / self.netScale) * self.netScale))
        Width_ = int(math.floor(math.ceil(Width / self.netScale) * self.netScale))

        if any([Height_ != Height, Width_ != Width]):
            tensor = F.pad(tensor, [0, Width_ - Width, 0, Height_ - Height])

        return tensor

    def getWeight(self, pathPreWeight: str = None):
        keyName = 'gNet'
        checkpoints = torch.load(pathPreWeight, map_location=torch.device('cpu'))
        try:
            weightDict = checkpoints[keyName]
        except Exception as e:
            weightDict = checkpoints['model_state_dict']
        return weightDict

    def setRequiresGrad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @torch.no_grad()
    def getFt(self, In, I0, I1, I2, t, flowNet):

        device0 = In.device
        N, C, H, W = I0.shape
        flowNet.eval()

        if (H * W) >= (1080 * 2048):
            down2x = partial(F.interpolate, scale_factor=0.5, mode='bilinear', align_corners=True)
            [In, I0, I1, I2] = [down2x(I) for I in [In, I0, I1, I2]]

        F0n = flowNet(I0, In)
        F01 = flowNet(I0, I1)
        a0 = (F01 + F0n) / 2.0
        b0 = (F01 - F0n) / 2.0
        F0t = a0 * (t ** 2) + b0 * t

        F12 = flowNet(I1, I2)
        F10 = flowNet(I1, I0)
        a1 = (F10 + F12) / 2.0
        b1 = (F10 - F12) / 2.0
        F1t = a1 * ((1 - t) ** 2) + b1 * (1 - t)

        if (H * W) >= (1080 * 2048):
            up2x = partial(F.interpolate, scale_factor=2.0, mode='bilinear', align_corners=True)
            [F0t, F1t] = [up2x(Fx * 2.0) for Fx in [F0t, F1t]]

        return F0t.to(device0).detach(), F1t.to(device0).detach()
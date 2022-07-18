import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.RAFT.update import BasicUpdateBlock, SmallUpdateBlock
from lib.RAFT.extractor import BasicEncoder, SmallEncoder
from lib.RAFT.corr import CorrBlock
from lib.RAFT.utils import coords_grid, upflow8
from collections import OrderedDict
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from lib.visualTool import visFlow
import argparse
from pathlib import Path


class RAFParam(object):
    def __init__(self):
        super(RAFParam, self).__init__()
        self.alternate_corr=False
        self.mixed_precision=False
        self.small=False


class RAFT(nn.Module):
    def __init__(self):
        super(RAFT, self).__init__()
        import warnings
        warnings.filterwarnings("ignore")
        self.args = RAFParam()

        if self.args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            self.args.corr_levels = 4
            self.args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            self.args.corr_levels = 4
            self.args.corr_radius = 4

        self.args.dropout = 0
        self.args.alternate_corr = False

        # feature network, context network, and update block
        if self.args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=self.args.dropout)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn='none', dropout=self.args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.args.dropout)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=self.args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        pathPreWeight = str(Path(__file__).parent.absolute() / Path('raft-things.pth'))
        self.initPreweight(pathPreWeight)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask, scale=8, ksize=3):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, C, H, W = flow.shape
        mask = mask.view(N, 1, ksize ** 2, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [ksize, ksize], padding=1)
        up_flow = up_flow.view(N, C, ksize ** 2, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, scale * H, scale * W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        # flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            # if up_mask is None:
            #     flow_up = upflow8(coords1 - coords0)
            # else:
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)

        return flow_up

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

def main():
    model = RAFT()
    model.cuda()
    model.eval()

    with torch.no_grad():
        image1 = cv2.imread(
            '/home/sensetime/data/VideoInterpolation/highfps/goPro/240fps/GoPro_public/test/GOPR0384_11_00/000009.png')
        image2 = cv2.imread(
            '/home/sensetime/data/VideoInterpolation/highfps/goPro/240fps/GoPro_public/test/GOPR0384_11_00/000017.png')

        image1 = torch.from_numpy(image1).float().permute([2, 0, 1]).unsqueeze(0)[:, [2, 1, 0], ...].cuda()
        image2 = torch.from_numpy(image2).float().permute([2, 0, 1]).unsqueeze(0)[:, [2, 1, 0], ...].cuda()

        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        visFlow(flow_up)


if __name__ == '__main__':
    main()
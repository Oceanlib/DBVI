import os

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import warnings

warnings.filterwarnings("ignore")



import ast
import argparse
from importlib import import_module

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=False, default='configTest', )
parser.add_argument('--initNode', type=str, required=True, default='XX.XX.XX.XX', help='Node for init')
parser.add_argument('--gpuList', type=str, required=False, default='{"PC":"0"}', help='gpuList for init')
parser.add_argument('--reuseGPU', type=int, required=False, default=0, help='reuseGPU or not')
parser.add_argument('--expName', type=str, required=False, default='DBVI', help='reuseGPU or not')

args = parser.parse_args()
gpuList = ast.literal_eval(args.gpuList)

config = import_module('configs.' + args.config)

reuseGPU = args.reuseGPU
if not reuseGPU:
    gpuList = None
cfg = config.configMain(gpuList, str(args.expName))


if 'GoPro' in cfg.dataset:
    from dataloader import GoPro as dataReader
elif 'Adobe' in cfg.dataset:
    from dataloader import Adobe as dataReader
else:
    from dataloader import XVFI as dataReader


import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

from collections import OrderedDict

from lib import fileTool as flLib
from lib import metrics as mrcLib
from lib import distTool as distLib

import torch
from torch import nn
from torch.distributed import init_process_group
from torch.backends import cudnn
from model.RRDBNet import IMRRDBNet

from apex.parallel import DistributedDataParallel
from apex.parallel import convert_syncbn_model
from configs.configTrain import configMain
from lib.RAFT.raft import RAFT as flowNet
from lib.visualTool import saveImg, visImg

class Tester(object):
    def __init__(self, cfg: configMain):
        if 0 == cfg.dist.gloRank:
            self.ckptBuffer = flLib.fileBuffer(cfg.state.maxSave)
            self.ckptBufferBest = flLib.fileBuffer(1)
        self.cfg: configMain = cfg
        self.codeCh = self.cfg.model.codeCh
        self.currtEpoch = 0
        self.currtIter = 0

        self.device0 = torch.device(f'cuda:{cfg.dist.locRank}')
        self.valiSampler, self.valiLoader = dataReader.creatValiLoader(cfg)

        self.gNet = IMRRDBNet(cfg=cfg).to(self.device0)
        self.gNet = convert_syncbn_model(self.gNet)
        self.gNet = DistributedDataParallel(self.gNet, delay_allreduce=False, gradient_average=True)

        self.flowNet = flowNet().eval().to(self.device0)
        self.metric = mrcLib.torchMetric(reduce='sum', device=self.device0).to(self.device0)
        self.trainInfoMsg, self.testInfoMsg = '', ''

    @torch.no_grad()
    def main(self):
        torch.cuda.empty_cache()
        self.gNet.eval()
        psnrSum = 0.0
        ssimSum = 0.0
        total = 0.0

        if 0 == cfg.dist.gloRank:
            pbar = tqdm(total=len(self.valiLoader), leave=True)
            psnrDict = OrderedDict()
            lpipsDict = OrderedDict()
            for keyIdx in range(self.cfg.numIter):
                psnrDict[f'It{keyIdx + 1}'] = mrcLib.AverageMeter()
                lpipsDict[f'It{keyIdx + 1}'] = mrcLib.AverageMeter()

            psnrDict['Total'] = mrcLib.AverageMeter()
            lpipsDict['Total'] = mrcLib.AverageMeter()

        for valIdx, (valBatchDict, watchList, gtList) in enumerate(self.valiLoader):
            I0 = valBatchDict['I0'].to(self.device0)
            I1 = valBatchDict['I1'].to(self.device0)

            if self.cfg.saveImg:
                I0Path = Path(self.cfg.outPath) / str(watchList[1][0])
                saveImg(I0, I0Path, isrgb=True)

                I1Path = Path(self.cfg.outPath) / str(watchList[2][0])
                saveImg(I1, I1Path, isrgb=True)

            for k in valBatchDict.keys():
                valBatchDict[k] = valBatchDict[k].to(self.device0)

            for tidx, t in enumerate(np.arange(1.0 / (self.cfg.numIter + 1.0), 1, 1.0 / (self.cfg.numIter + 1.0))):
                output = self.gNet(batchDict=valBatchDict, flowNet=self.flowNet, t=t)
                fake = output[-1]
                gt = valBatchDict[f'It{tidx + 1}']

                if self.cfg.saveImg:
                    outPath = Path(self.cfg.outPath) / str(gtList[tidx][0]).replace('.png', '_inter.png')
                    saveImg(fake, outPath, isrgb=True)

                psnr, ssim, N = self.metric(fake.detach(), gt.detach())
                psnr = distLib.reduceTensorSum(psnr).detach().cpu().item()
                ssim = distLib.reduceTensorSum(ssim).detach().cpu().item()

                psnrSum += psnr
                ssimSum += ssim

                N = torch.tensor(N, dtype=torch.float32, device=self.device0)
                total += (distLib.reduceTensorSum(N)).detach().cpu().item()

                if 0 == self.cfg.dist.gloRank:
                    psnrDict[f'It{tidx + 1}'].update(psnr / N)
                    psnrDict['Total'].update(psnr)

            psnrAvg = psnrSum / total
            ssimAvg = ssimSum / total
            if 0 == cfg.dist.gloRank:
                description = [f'Test: ']
                description.append(f"psnr={psnrAvg :.5f}  ")
                description.append(f"ssim={ssimAvg:.5f}  ")
                pbarMsg = ''.join(description).ljust(50, ' ')
                pbar.set_description(pbarMsg)
                pbar.update(1)

        psnrAvg = psnrSum / total
        ssimAvg = ssimSum / total
        if 0 == cfg.dist.gloRank:
            pbar.clear()
            pbar.close()
            for k, v in psnrDict.items():
                psnrDict[k] = v.avg if k != 'Total' else v.sum / total
            for k, v in lpipsDict.items():
                lpipsDict[k] = v.avg if k != 'Total' else v.sum / total

            description = [f"Testing :"]
            description.append(f"psnr={psnrAvg:.5f}  ")
            description.append(f"ssim={ssimAvg:.5f}  ")
            infoMsg = ''.join(description)
            self.testInfoMsg = infoMsg
            self.info("".center(100, '-'))
            self.info(" ")
            self.info(f"{self.trainInfoMsg} || {self.testInfoMsg}")
            self.info(" ")
            self.info("".center(100, '-'))

        self.gNet.train()
        return psnrAvg

    def creatNet(self, net: nn.Module, dist=True, requireGrad=True):
        net = net(cfg=self.cfg).to(self.device0)
        self.setRequiresGrad([net], requireGrad)
        if dist:
            net = convert_syncbn_model(net)
            net = DistributedDataParallel(net, delay_allreduce=False, gradient_average=True)
        return net

    def saveState(self, psnr):
        if 0 == cfg.dist.gloRank:
            if psnr >= self.maxPSNR:
                self.maxPSNR = psnr
                savePath = str(Path(cfg.path.ckpts) /
                               f'best_Epoch{self.currtEpoch}_Iter{self.currtIter}_PSNR{psnr:.3f}.pth')
                self.ckptBufferBest(savePath)
            else:
                savePath = str(Path(cfg.path.ckpts) /
                               f'Epoch{self.currtEpoch}_Iter{self.currtIter}_PSNR{psnr:.3f}.pth')
                self.ckptBuffer(savePath)

            if hasattr(self.gNet, 'module'):  # parallel may add 'module' in name
                weight = self.gNet.module.state_dict()
            else:
                weight = self.gNet.state_dict()

            if cfg.dist.useApex:
                stateDict = {'last_epoch': self.currtEpoch,
                             'last_Iter': self.currtIter,
                             'gNet': weight,
                             'state_dict_Optimizer': self.optim.state_dict(),
                             'state_dict_Scheduler': self.scheduler.state_dict(),
                             'maxPSNR': self.maxPSNR
                             }
            else:
                stateDict = {'last_epoch': self.currtEpoch,
                             'last_Iter': self.currtIter,
                             'gNet': weight,
                             'state_dict_Optimizer': self.optim.state_dict(),
                             'state_dict_Scheduler': self.scheduler.state_dict(),
                             'maxPSNR': self.maxPSNR
                             }

            torch.save(stateDict, savePath)
            self.info("".ljust(100, '.'))
            self.info('Saving model in ' + savePath)
            self.info('\n')

    def updateLR(self, metric):
        # self.info(f'mertic={metric}')
        self.scheduler.step(metrics=metric)
        return self.optim.param_groups[0]['lr']

    def setRequiresGrad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def info(self, msg: str):
        if 0 == cfg.dist.gloRank:
            cfg.logger.info(msg)

    def writeScalar(self, tag, scalar_value, global_step):
        if 0 == cfg.dist.gloRank:
            cfg.state.sumWriter.add_scalar(tag, scalar_value, global_step)

    def writeScalars(self, main_tag, tag_scalar_dict, global_step):
        if 0 == cfg.dist.gloRank:
            cfg.state.sumWriter.add_scalars(main_tag, tag_scalar_dict, global_step)

    def writeImg(self, tag, img_tensor, global_step=None):
        if 0 == cfg.dist.gloRank:
            cfg.state.sumWriter.add_image(tag, img_tensor, global_step)

    def writeImgs(self, tag, img_tensor, global_step=None):
        if 0 == cfg.dist.gloRank:
            cfg.state.sumWriter.add_images(tag, img_tensor, global_step)

    def writeVideo(self, tag, vid_tensor: torch.Tensor, global_step=None):
        if 0 == cfg.dist.gloRank:
            vid_tensor = ((vid_tensor + 1.0) / 2.0).clamp(0, 1.0)
            cfg.state.sumWriter.add_video(tag, vid_tensor, global_step)


def preInit():
    seed = cfg.state.randomSeed + cfg.dist.gloRank

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.cuda.set_device(cfg.dist.locRank)
    assert cudnn.enabled, "Amp requires cudnn backend to be enabled."
    torch.backends.cudnn.benchmark = True
    init_process_group(backend='nccl',
                       init_method='tcp://' + args.initNode + ':5802',
                       world_size=cfg.dist.wordSize,
                       rank=cfg.dist.gloRank)
    distLib.synchronize()


if __name__ == '__main__':
    preInit()
    tester = Tester(cfg)
    tester.main()

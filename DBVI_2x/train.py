import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ["IMAGEIO_FFMPEG_EXE"] = "~/anaconda3/envs/torch18/bin/ffmpeg"
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import warnings

warnings.filterwarnings("ignore")
import ast
import argparse
from importlib import import_module

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=False, default='configTrain')
parser.add_argument('--initNode', type=str, required=True, default='XX.XX.XX.XX', help='Node for init')
parser.add_argument('--gpuList', type=str, required=False, default='{"PC":"0"}', help='gpuList for init')
parser.add_argument('--reuseGPU', type=int, required=False, default=0, help='reuseGPU or not')
parser.add_argument('--expName', type=str, required=False, default='DBVI_2x', help='name of experiment')

args = parser.parse_args()
gpuList = ast.literal_eval(args.gpuList)

config = import_module('configs.' + args.config)

reuseGPU = args.reuseGPU
if not reuseGPU:
    gpuList = None
cfg = config.configMain(gpuList, str(args.expName))
import time
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import traceback

from collections import OrderedDict
from dataloader.Vimeo import creatTrainLoader
from dataloader.Vimeo import creatValiLoader

from lib import fileTool as flLib
from lib import metrics as mrcLib
from lib import distTool as distLib
from lib import lossTool as lossLib
from lib.dlTool import getScheduler, getOptimizer

import torch
from torch import nn
from torch.distributed import init_process_group
from torch.backends import cudnn
from model.RRDBNet import IMRRDBNet

from apex.parallel import DistributedDataParallel
from apex.parallel import convert_syncbn_model
from configs.configTrain import configMain
from lib.RAFT.raft import RAFT


class Trainer(object):
    def __init__(self, cfg: configMain):
        if 0 == cfg.dist.gloRank:
            self.ckptBuffer = flLib.fileBuffer(cfg.state.maxSave)
            self.ckptBufferBest = flLib.fileBuffer(1)
        self.cfg: configMain = cfg

        self.currtEpoch = 0
        self.currtIter = 0

        self.device0 = torch.device(f'cuda:{cfg.dist.locRank}')
        self.trainSampler, self.trainLoader = creatTrainLoader(cfg)
        self.valiSampler, self.valiLoader = creatValiLoader(cfg)

        self.gNet = IMRRDBNet(cfg=cfg).to(self.device0)
        self.gNet = convert_syncbn_model(self.gNet)
        self.gNet = DistributedDataParallel(self.gNet, delay_allreduce=False, gradient_average=True)
        self.setRequiresGrad([self.gNet], requires_grad=True)

        self.flowNet = RAFT().eval().to(self.device0)
        self.setRequiresGrad([self.flowNet], requires_grad=False)

        self.optim, self.maxPSNR = getOptimizer(self.gNet, cfg)

        self.initState()
        self.scheduler = getScheduler(self.optim, epoch=self.currtEpoch)

        self.lr = self.optim.param_groups[0]['lr']
        self.criterion = lossLib.totalLoss()
        self.metric = mrcLib.torchMetric(reduce='sum', device=self.device0).to(self.device0)
        self.trainInfoMsg, self.testInfoMsg = '', ''

    def main(self):
        # init-------------------------------------------
        psnr = self.validation()
        # self.saveState(psnr=psnr)
        try:
            lastEpoch = self.currtEpoch
            for epoch in range(lastEpoch, cfg.train.maxEpoch):
                torch.cuda.empty_cache()
                self.lr = self.updateLR()
                self.trainEpoch()
                if self.currtEpoch % cfg.train.snapShot == 0:
                    psnr = self.validation()
                    self.saveState(psnr)
                self.currtEpoch += 1

        except Exception as e:
            self.info(traceback.format_exc(limit=100))
            self.saveState(psnr=self.maxPSNR)
            path = str(Path(cfg.path.ckpts) / f'epochAuto{self.currtEpoch}.pth')
            self.info('AutoSaving model in ' + path + '\n')

    def trainEpoch(self):
        torch.cuda.empty_cache()
        self.trainSampler.set_epoch(self.currtEpoch)
        self.gNet.train()

        if 0 == cfg.dist.gloRank:
            self.info(f'{args.expName} Epoch:{self.currtEpoch} lr:{self.lr}'.center(100, '='))
            movingDict = OrderedDict()
            for lossName in self.criterion.lossNames:
                movingDict[lossName] = mrcLib.AverageMeter()
            movingDict['Total'] = mrcLib.AverageMeter()
            pbar = tqdm(total=len(self.trainLoader), leave=False)
            start = time.time()

        for bIdx, batchDict in enumerate(self.trainLoader):

            self.optim.zero_grad()

            for k, v in batchDict.items():
                batchDict[k] = v.to(self.device0)

            gts = []
            for s in self.cfg.model.scale:
                gts.append(batchDict[f'It'].to(self.device0).detach())

            rgbs = self.gNet(batchDict=batchDict, flowNet=self.flowNet)
            lossAvg = self.criterion(rgbs=rgbs, gts=gts)

            self.optim.zero_grad()
            lossAvg.backward()
            self.optim.step()

            for k, v in self.criterion.lossDict.items():
                v = distLib.reduceTensorMean(v)
                vitem = v.detach().cpu().item()
                if 0 == cfg.dist.gloRank:
                    movingDict[k].update(vitem)

            if 0 == cfg.dist.gloRank:

                description = [f'INFO: :Training: ']
                for k, v in movingDict.items():
                    description.append(f'{k}={v.avg:.5f}  ')
                pbarMsg = ''.join(description).ljust(50, ' ')
                pbar.set_description(pbarMsg)
                pbar.update(1)

                self.currtIter += 1

        if 0 == cfg.dist.gloRank:
            pbar.clear()
            pbar.close()
            torch.cuda.current_stream().synchronize()
            end = time.time()
            for k, v in movingDict.items():
                movingDict[k] = v.avg
            description = [f"Training: time:{(end - start) / 60.0:.2f}min  "]
            for k, v in movingDict.items():
                description.append(f'{k}={v:.5f}  ')
            infoMsg = ''.join(description)
            self.trainInfoMsg = infoMsg
            self.info(self.trainInfoMsg + '\n')

            self.writeScalars('losses', movingDict, self.currtIter)
            self.writeScalar('lr', self.lr, self.currtIter)

    @torch.no_grad()
    def validation(self):
        torch.cuda.empty_cache()
        self.gNet.eval()
        psnrSum = 0.0
        ssimSum = 0.0
        total = 0.0

        if 0 == cfg.dist.gloRank:
            pbar = tqdm(total=len(self.valiLoader), leave=True)
            psnrDict = OrderedDict()
            lpipsDict = OrderedDict()
            psnrDict['It'] = mrcLib.AverageMeter()
            lpipsDict['It'] = mrcLib.AverageMeter()

            psnrDict['Total'] = mrcLib.AverageMeter()
            lpipsDict['Total'] = mrcLib.AverageMeter()

        for valIdx, (valBatchDict, watchList, gtList) in enumerate(self.valiLoader):
            for k in valBatchDict.keys():
                valBatchDict[k] = valBatchDict[k].to(self.device0)
            output = self.gNet(batchDict=valBatchDict, flowNet=self.flowNet, t=0.5)
            fake = output[-1]
            fake = (fake + 1.0) / 2.0
            gt = valBatchDict['It']
            gt = (gt + 1.0) / 2.0
            psnr, ssim, N = self.metric(fake.detach(), gt.detach(), ifssim=False)

            psnr = distLib.reduceTensorSum(psnr).detach().cpu().item()
            ssim = distLib.reduceTensorSum(ssim).detach().cpu().item()

            psnrSum += psnr
            ssimSum += ssim

            N = torch.tensor(N, dtype=torch.float32, device=self.device0)
            N = (distLib.reduceTensorSum(N)).detach().cpu().item()
            total += N

            if 0 == self.cfg.dist.gloRank:
                psnrDict[f'It'].update(psnr / N)

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

    def initState(self):
        if cfg.resume:
            weightPath = str(cfg.path.ckpt)
            self.info('Loading from ' + weightPath)
            checkpoints = torch.load(weightPath, map_location=torch.device('cpu'))
            try:
                self.maxPSNR = checkpoints['maxPSNR']
                self.currtEpoch = checkpoints['last_epoch']
                self.currtIter = checkpoints['last_Iter']
                self.optim.load_state_dict(checkpoints['state_dict_Optimizer'])

                if cfg.optim.forceLrInit is not None:
                    for param_group in self.optim.param_groups:
                        param_group['lr'] = cfg.optim.forceLrInit
                        param_group.setdefault('initial_lr', param_group['lr'])

                self.info(f'currtEpoch={self.currtEpoch}, currtIter={self.currtIter}, maxPSNR={self.maxPSNR}')
            except Exception as e:
                self.info(traceback.format_exc(limit=100))
                self.info('loading state dict for optimazer failed'.center(100, '*'))
        else:
            self.info('training from scratch')

    def creatNet(self, net: nn.Module, dist=True, requireGrad=True):
        net = net(cfg=self.cfg).to(self.device0)
        self.setRequiresGrad([net], requireGrad)
        if dist:
            net = convert_syncbn_model(net)
            net = DistributedDataParallel(net, delay_allreduce=False, gradient_average=True)
        return net

    def saveState(self, psnr):
        if 0 == cfg.dist.gloRank:
            #####################################################
            if psnr >= self.maxPSNR:
                #####################################################
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
            # self.info("".ljust(100, '.'))
            # self.info('End'.center(100, '='))
            self.info('\n')

    def updateLR(self, metric=0):
        self.scheduler.step()
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
                       init_method='tcp://' + args.initNode + ':5801',
                       world_size=cfg.dist.wordSize,
                       rank=cfg.dist.gloRank)
    distLib.synchronize()


if __name__ == '__main__':
    preInit()
    trainer = Trainer(cfg)
    trainer.main()

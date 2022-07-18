import os
import logging
import torch
from datetime import datetime
from lib import fileTool as FT
from pathlib import Path
from tensorboardX import SummaryWriter
import re
import numpy as np


class _Dist(object):
    def __init__(self):
        super(_Dist, self).__init__()
        self.isDist = True
        self.nodeName = self.getDist('SLURMD_NODENAME', 'PC')
        self.wordSize = self.getDist('SLURM_NTASKS', 1)
        self.nodeID = self.getDist('SLURM_NODEID', 0)
        self.gloRank = self.getDist('SLURM_PROCID', 0)
        self.locRank = self.getDist('SLURM_LOCALID', 0)

        self.useApex = False
        self.apexLevel = 'O0'

        if any(['PC' == self.nodeName, 1 == self.wordSize]):
            self.isDist = False

    def getDist(self, key: str, default):
        out = os.environ[key] if key in os.environ else default
        if isinstance(default, int):
            out = int(out)
        elif isinstance(default, str):
            out = str(out)
        return out

    def __repr__(self):
        return 'Dist'


_dist = _Dist()


class configMain(object):
    def __init__(self, gpuList=None, expName='Hyper'):
        super(configMain, self).__init__()
        if gpuList is not None:
            self.gpu = gpuList[_dist.nodeName]
            os.environ['CUDA_VISIBLE_DEVICES'] = gpuList[_dist.nodeName]  # crucial

        # Vimeo, ucf101, davis, snufilm-easy, snufilm-medium,snufilm-hard,snufilm-extreme
        self.dataset = 'snufilm-hard'

        self.resume = True
        self.expName = '2x_Vimeo' if self.resume else expName
        self.preTrained = 'train_on_vimeo_sep_Iter0.pth' if self.resume else None
        self.numIter = 1

        self.saveImg = False
        self.outPath = f'Path/to/images/'  # end with /

        self.normal = True

        self.dist = _dist
        self.path = _Path(resume=self.resume, expName=self.expName, cpkt=self.preTrained)
        self.model = _Model()
        self.train = _Train()
        self.test = _Test()
        self.optim = _Optim()
        self.lrSch = _LrScheduler()
        self.logger = self.logInit()

        self.state = _State(gpuList, self.path.ckpt, self.path.events, self.logger, self.resume)

        self.record()

    def logInit(self):
        if 0 == self.dist.gloRank:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(' ')
            logfile = str(Path(self.path.exp) / 'log.txt')
            fh = logging.FileHandler(logfile, mode='a')
            formatter = logging.Formatter(fmt="%(asctime)s: %(message)s", datefmt="%m-%d-%H-%M")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            return logger

    def record(self):
        def logclass(obj, logger):
            args = vars(obj)
            for k, v in sorted(args.items()):
                if '_' in type(v).__name__:
                    logclass(v, logger)
                else:
                    logger_.info('{}.{} = {}'.format(obj.__repr__(), k, v))

        if 0 == self.dist.gloRank:
            logger_ = logging.getLogger('Config')
            logging.basicConfig(level=logging.INFO)
            path = str(Path(self.path.exp) / Path('config.txt'))
            fh = logging.FileHandler(path, mode='w')
            formatter = logging.Formatter(fmt="%(asctime)s: %(message)s", datefmt="%m-%d-%H-%M")
            fh.setFormatter(formatter)
            logger_.addHandler(fh)
            logger_.info('-------------Config-------------------')
            logclass(self, logger_)
            logger_.info('--------------End---------------------')

    def __repr__(self):
        return 'configMain'


class _Path(object):
    """"
    Path related parameters
    """

    def __init__(self, resume=False, expName=None, cpkt=None):
        super(_Path, self).__init__()
        # visible for all process---------------------------------------------------------------------------
        self.resume = resume

        self.output = './output'

        self.exp = None
        self.ckpts = None
        self.events = None
        self.ckpt = None

        if self.resume:
            self.exp = str(Path(self.output) / Path(expName))
            self.ckpts = str(Path(self.exp) / 'ckpts')
            self.events = str(Path(self.exp) / 'events')
            self.img = str(Path(self.exp) / 'img')
            self.ckpt = str(Path(self.ckpts) / cpkt)
            if 0 == _dist.gloRank:
                assert all([Path(self.ckpt).is_file(),
                            Path(self.exp).is_dir(),
                            Path(self.ckpts).is_dir(),
                            Path(self.events).is_dir()]), 'path Error'
        # visible for all process-----------------------------------------------------------------------
        else:
            if 0 == _dist.gloRank:
                now = datetime.now().strftime("%Y%m%d%H%M")
                self.exp = str(Path(self.output) / '{}_{}'.format(expName, now))
                self.ckpts = str(Path(self.exp) / 'ckpts')
                self.events = str(Path(self.exp) / 'events')
                self.img = str(Path(self.exp) / 'img')
                FT.mkPath(self.exp)
                FT.mkPath(self.ckpts)
                FT.mkPath(self.events)
                FT.mkPath(self.img)

                assert all([Path(self.exp).is_dir(),
                            Path(self.ckpts).is_dir(),
                            Path(self.events).is_dir()]), 'path Error'

    def __repr__(self):
        return 'Path'


class _Model(object):
    """"
    Training related parameters
    """

    def __init__(self):
        super(_Model, self).__init__()
        self.name = 'RRDB'
        self.initType = 'kaiming'  # 'xavier', 'kaiming', 'orthogonal', 'default'
        # self.codeCh = 3
        self.codeCh = 0
        # self.scale = [8, 4, 2, 1]
        self.scale = [1, 1, 1, 1, 1, 1]
        self.resProb = False
        # self.scale = [1, 1, 1, 1]

    def __repr__(self):
        return 'Model'


class _Train(object):
    """"
    Training related parameters
    """

    def __init__(self):
        super(_Train, self).__init__()
        # all batch are per gpu
        # self.monthBatch = 640 if _dist.isDist else 640  # large batch for sub dataset
        self.batchPerGPU = 2 if _dist.isDist else 1  # large batch for sub dataset

        self.dayBatch = 2 if _dist.isDist else 1  # real batch for update parameters
        self.searchBatch = 10 if _dist.isDist else 10  # batch for search noise
        self.nNoisePerImg = 120

        self.nWorkers = 4 if _dist.isDist else 4

        self.maxEpoch = 20000

        self.dayEpoch = 1
        self.snapShot = 5
        self.size = (256, 256)

    def __repr__(self):
        return 'Train'


class _Test(object):
    """"
    Testing related parameters
    """

    def __init__(self):
        super(_Test, self).__init__()
        self.batchPerGPU = 1 if _dist.isDist else 1
        self.nWorkers = 4 if _dist.isDist else 0
        self.batch = self.batchPerGPU * _dist.wordSize
        # self.size = (700, 1000)

        self.size = (256, 256)

    def __repr__(self):
        return 'Test'


class _Optim(object):
    """"
    optimizier related parameters
    """

    def __init__(self):
        super(_Optim, self).__init__()
        self.policy = 'adam'
        self.lrInit = 1e-4
        self.forceLrInit = None
        self.betas = (0.9, 0.99)
        self.weightDecay = 0
        self.momentum = 0.9

    def __repr__(self):
        return 'Optim'


class _LrScheduler(object):
    """"
    LrSch related parameters
    """

    def __init__(self):
        super(_LrScheduler, self).__init__()
        self.policy = 'plateau'

        self.factor = 0.5
        self.patience = 1

        self.mode = 'min'

    def __repr__(self):
        return 'LrSch'


class _State(object):
    """"
    current state related parameters(workers, random seed, etc)
    """

    def __init__(self, gpuList, pathCkpt, pathEvents, logger, resume):
        super(_State, self).__init__()
        self.randomSeed = 1234
        self.netCheck = False
        self.maxSave = 2  # number of saved checkpoints
        self.nGPUs = torch.cuda.device_count()
        if 0 == _dist.gloRank:
            if resume:
                currtIter = int(re.search('Iter(\d+)', pathCkpt).group(1))

                self.sumWriter = SummaryWriter(pathEvents, purge_step=currtIter)
                logger.info(f'loading from {pathEvents}'.center(50, '-'))
            else:
                self.sumWriter = SummaryWriter(pathEvents)
                logger.info('currtIter=0 and currtEpoch=0'.center(100, '-'))

    def __repr__(self):
        return 'expState'

if __name__ == '__main__':
    cfg = configMain(gpuList='0')

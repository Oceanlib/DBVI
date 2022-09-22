import cv2
import torch.nn.functional as F

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from dataloader import cvtransform
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
from dataloader.dataloaderBase import DistributedSamplerVali
from configs.configTrain import configMain
import pickle
import lmdb
from lib.visualTool import visImg
import torch
import numpy as np


class Train(data.Dataset):
    def __init__(self, cfg: configMain):
        self.LMDB = './dataset/X4K1000FPS/train_lmdb/data.mdb'
        self.Sample = './dataset/X4K1000FPS/train_lmdb/sample.pkl'

        self.numIter = cfg.numIter

        with open(self.Sample, 'rb') as fs:
            self.Sample = pickle.load(fs)

        self.length = len(self.Sample)  #1500

        self.transforms = cvtransform.Compose([
            cvtransform.RandomCrop(cfg.train.size),
            cvtransform.RandomHorizontalFlip(0.5),
            cvtransform.RandomHVerticalFlip(0.5),
            cvtransform.ColorJitter(0.05, 0.05, 0.05, 0.05),
            cvtransform.ToTensor()
        ])
        self.env = None
        self.txn = None
        self.outKeys = ['In', 'I0', 'It_1', 'I1', 'I2']

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        if any([self.txn is None, self.env is None]):
            self.env = lmdb.open(self.LMDB, subdir=False, readonly=True, lock=False, readahead=False,
                                 meminit=False)
            self.txn = self.env.begin(write=False)

        sampleKeys = self.Sample[idx]

        In = pickle.loads(self.txn.get(sampleKeys['In'].encode('ascii')))
        I0 = pickle.loads(self.txn.get(sampleKeys['I0'].encode('ascii')))
        I1 = pickle.loads(self.txn.get(sampleKeys['I1'].encode('ascii')))
        I2 = pickle.loads(self.txn.get(sampleKeys['I2'].encode('ascii')))

        t = np.random.randint(1, self.numIter + 1)
        It = pickle.loads(self.txn.get(sampleKeys[f'It{t}'].encode('ascii')))
        t = torch.tensor(t / float(self.numIter + 1.0), dtype=torch.float32)

        if np.random.rand() > 0.5:
            valueList = [In, I0, It, I1, I2]
        else:
            valueList = [I2, I1, It, I0, In]
            t = 1 - t

        valueList = self.transforms(valueList)


        valueList = valueList

        outDict = {'t': t}
        for key, value in zip(self.outKeys, valueList):
            outDict[key] = value
        return outDict

    def __len__(self):
        return self.length

    def close(self):
        if self.env is not None:
            self.env.close()
            self.txn = None
            self.env = None


def creatTrainLoader(cfg: configMain):
    dataset = Train(cfg)

    sampler = DistributedSampler(dataset, num_replicas=cfg.dist.wordSize, rank=cfg.dist.gloRank)

    loader = data.DataLoader(dataset=dataset, batch_size=cfg.train.batchPerGPU,
                             shuffle=False, num_workers=4, pin_memory=True,  # False if memory is not enough
                             drop_last=True, sampler=sampler)
    return sampler, loader


class Test(data.Dataset):
    def __init__(self, cfg: configMain):
        super(Test, self).__init__()
        self.Sample = './datasets/X4K1000FPS/test_lmdb/sample.pkl'
        self.LMDB = './datasets/X4K1000FPS/test_lmdb/data.mdb'
        self.numIter = cfg.numIter

        with open(self.Sample, 'rb') as fs:
            self.Sample = pickle.load(fs)

        self.length = len(self.Sample)

        self.transforms = cvtransform.Compose([
            cvtransform.ToTensor()
        ])
        self.env = None
        self.txn = None
        self.outKeys = ['In', 'I0', 'I1', 'I2', 'It1', 'It2', 'It3', 'It4', 'It5', 'It6', 'It7']

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        if any([self.txn is None, self.env is None]):
            self.env = lmdb.open(self.LMDB, subdir=False, readonly=True, lock=False, readahead=False,
                                 meminit=False)
            self.txn = self.env.begin(write=False)

        sampleKeys = self.Sample[idx]

        In = pickle.loads(self.txn.get(sampleKeys['In'].encode('ascii')))
        I0 = pickle.loads(self.txn.get(sampleKeys['I0'].encode('ascii')))
        I1 = pickle.loads(self.txn.get(sampleKeys['I1'].encode('ascii')))
        I2 = pickle.loads(self.txn.get(sampleKeys['I2'].encode('ascii')))

        valueList = [In, I0, I1, I2]
        watchList = [sampleKeys['In'], sampleKeys['I0'], sampleKeys['I1'], sampleKeys['I2']]
        gtList = []

        for i in range(self.numIter):
            It = pickle.loads(self.txn.get(sampleKeys[f'It{i + 1}'].encode('ascii')))
            valueList.append(It)
            gtList.append(sampleKeys[f'It{i + 1}'])
        valueList = self.transforms(valueList)

        outDict = {}
        for key, value in zip(self.outKeys, valueList):
            outDict[key] = value

        return outDict, watchList, gtList

    def __len__(self):
        return self.length


def creatValiLoader(cfg: configMain):
    dataset = Test(cfg)

    # if cfg.dist.isDist:
    sampler = DistributedSamplerVali(dataset, num_replicas=cfg.dist.wordSize, rank=cfg.dist.gloRank)

    loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                             num_workers=2, pin_memory=False,  # False if memory is not enough
                             drop_last=False, sampler=sampler)
    return sampler, loader

from pathlib import Path

import re
import pickle
from lib import fileTool as FT
from lib.dataUtils import sortFunc


def extractKey(name: str):
    parts = Path(name).parts
    key = f'{parts[-2]}/{parts[-1]}'
    return key


def sortFunc(name: str):
    idx = re.search(r'(\d*).png', str(name)).group(1)
    return int(idx)


def genSampleTrain(srcDir, samplePath, numIter=7):
    idxV = [0, (numIter + 1) * 2, (2 * numIter + 2) * 2, (3 * numIter + 3) * 2]
    idxT = [numIter * 2 + 2 + i * 2 + 2 for i in range(numIter)]
    idxVT = idxV + idxT

    keyV = ['In', 'I0', 'I1', 'I2']
    keyT = [f'It{i + 1}' for i in range(numIter)]
    keyVT = keyV + keyT

    allSubDirs = FT.getSubDirs(srcDir)
    allSamples = []
    for subDir in allSubDirs:
        allFrames = FT.getAllFiles(subDir, 'png')
        allFrames.sort(key=sortFunc)

        # for startIdx in range(len(allFrames)):
        # for startIdx in [0, 3, 8, 11, 14, 16]:
        for startIdx in [0, 3, 8, 11, 16]:
            sampleDict = {}
            endIdx = startIdx + 49
            asample = allFrames[startIdx:endIdx]
            asample = [extractKey(i) for i in asample]
            for idx, key in zip(idxVT, keyVT):
                sampleDict[key] = asample[idx]
            allSamples.append(sampleDict)

    with open(samplePath, 'wb') as f:
        pickle.dump(allSamples, f)


if __name__ == '__main__':
    numIter = 7
    srcDir = '/data/dataset/X4K1000FPS/train'
    samplePath = '/data/dataset/X4K1000FPS/train_lmdb/sample.pkl'

    genSampleTrain(srcDir=srcDir, samplePath=samplePath, numIter=numIter)

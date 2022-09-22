from pathlib import Path

import pickle
from lib import fileTool as FT


def extractKey(name: str):
    parts = Path(name).parts
    key = f'{parts[-2]}/{parts[-1]}'
    return key


def genSampleTest(srcDir, samplePath, numIter):
    allSubDirs = FT.getSubDirs(srcDir)
    allSamples = []
    keyV = ['In', 'I0', 'I1', 'I2']
    keyT = [f'It{i + 1}' for i in range(numIter)]
    keyVT = keyV + keyT

    for subDir in allSubDirs:
        sampleDict={}
        allFrames = FT.getAllFiles(subDir, 'png')

        for fname in allFrames:
            for akey in keyVT:
                if f'{akey}.png' in fname:
                    sampleDict[akey] = extractKey(fname)
                    break
        allSamples.append(sampleDict)

    with open(samplePath, 'wb') as f:
        pickle.dump(allSamples, f)


if __name__ == '__main__':
    numIter = 7
    srcDir = './datasets/X4K1000FPS/test_frames/'
    samplePath = './datasets/X4K1000FPS/X4k_lmdb//sample.pkl'

    genSampleTest(srcDir=srcDir, samplePath=samplePath, numIter=numIter)

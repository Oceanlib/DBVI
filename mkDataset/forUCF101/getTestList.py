from pathlib import Path

import pickle
from lib import fileTool as FT


def extractKey(name: str):
    parent = Path(name).parent.name
    fileName = Path(name).name
    key = f'{parent}/{fileName}'
    return key


def genSampleTest(srcDir, samplePath, numIter):
    allSubDirs = FT.getSubDirs(srcDir)
    allSamples = []

    for subDir in allSubDirs:
        sampleDict={}
        allFrames = FT.getAllFiles(subDir, 'png')
        for fname in allFrames:
            if 'frame0.png' in fname:
                sampleDict['In'] = extractKey(fname)
            if 'frame1.png' in fname:
                sampleDict['I0'] = extractKey(fname)
            if 'frame2.png' in fname:
                sampleDict['I1'] = extractKey(fname)
            if 'frame3.png' in fname:
                sampleDict['I2'] = extractKey(fname)
            if 'framet.png' in fname:
                sampleDict['It'] = extractKey(fname)
        allSamples.append(sampleDict)

    with open(samplePath, 'wb') as f:
        pickle.dump(allSamples, f)


if __name__ == '__main__':
    numIter = 1
    srcDir = '/Path/to/ucf101/'
    samplePath = 'Path/to/ucf101/ucf101_lmdb/sample.pkl'

    genSampleTest(srcDir=srcDir, samplePath=samplePath, numIter=numIter)

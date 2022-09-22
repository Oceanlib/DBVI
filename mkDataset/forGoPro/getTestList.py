from pathlib import Path

import re
import pickle
from lib import fileTool as FT
from lib.dataUtils import sortFunc


def extractKey(name: str):
    parts = Path(name).parts
    key = f'{parts[-3]}/{parts[-2]}/{parts[-1]}'
    return key


def sortFunc(name: str):
    idx = re.search(r'(\d*).png', str(name)).group(1)
    return int(idx)


def genSampleTest(srcDir, samplePath, numIter):
    idxV = [0, numIter + 1, 2 * numIter + 2, 3 * numIter + 3]
    idxT = [numIter + 1 + i + 1 for i in range(numIter)]
    idxVT = idxV + idxT

    keyV = ['In', 'I0', 'I1', 'I2']
    keyT = [f'It{i + 1}' for i in range(numIter)]
    keyVT = keyV + keyT

    allSubDirs = FT.getSubDirs(srcDir)
    # allSamplesNames = []
    allSamples = []
    for subDir in allSubDirs:
        allFrames = FT.getAllFiles(subDir, 'png')
        allFrames.sort(key=sortFunc)

        for startIdx in range(0, len(allFrames), numIter+1):
            sampleDict = {}
            endIdx = startIdx + numIter * 3 + 4
            if endIdx <= len(allFrames):
                asample = allFrames[startIdx:endIdx]
                asample = [extractKey(i) for i in asample]
                for idx, key in zip(idxVT, keyVT):
                    sampleDict[key] = asample[idx]
                allSamples.append(sampleDict)

            # elif endIdx != len(allFrames) + 1:
            #     asample = allFrames[-(numIter * 3 + 4)::]
            #     asample = [extractKey(i) for i in asample]
            #     for idx, key in zip(idxVT, keyVT):
            #         sampleDict[key] = asample[idx]
            #     allSamples.append(sampleDict)
            #     break
            # else:
            #     break
        with open(samplePath, 'wb') as f:
            pickle.dump(allSamples, f)


if __name__ == '__main__':
    numIter = 7
    srcDir = 'Path/to/GoPro/test'
    samplePath = 'Path/to/GoPro/gopro_test_lmdb/sample.pkl'

    genSampleTest(srcDir=srcDir, samplePath=samplePath, numIter=numIter)

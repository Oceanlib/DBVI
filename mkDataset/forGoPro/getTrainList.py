import re
import pickle
import lib.fileTool as FT
from pathlib import Path


def extractKey(name: str):
    parts = Path(name).parts
    key = f'{parts[-3]}/{parts[-2]}/{parts[-1]}'
    return key


def sortFunc(name: str):
    idx = re.search(r'(\d*).png', str(name)).group(1)
    return int(idx)


def genSampleTrain(srcDir, samplePath, numIter=7):
    idxV = [0, numIter + 1, 2 * numIter + 2, 3 * numIter + 3]
    idxT = [numIter + 1 + i + 1 for i in range(numIter)]
    idxVT = idxV + idxT

    keyV = ['In', 'I0', 'I1', 'I2']
    keyT = [f'It{i + 1}' for i in range(numIter)]
    keyVT = keyV + keyT

    allSubDirs = FT.getSubDirs(srcDir)
    allSamples = []
    for subDir in allSubDirs:
        allFrames = FT.getAllFiles(subDir, 'png')
        allFrames.sort(key=sortFunc)

        for startIdx in range(len(allFrames)):
            sampleDict = {}
            endIdx = startIdx + numIter * 3 + 4
            if endIdx <= len(allFrames):
                asample = allFrames[startIdx:endIdx]
                asample = [extractKey(i) for i in asample]
                for idx, key in zip(idxVT, keyVT):
                    sampleDict[key] = asample[idx]
                allSamples.append(sampleDict)

            elif endIdx != len(allFrames) + 1:
                asample = allFrames[-(numIter * 3 + 4)::]
                asample = [extractKey(i) for i in asample]
                for idx, key in zip(idxVT, keyVT):
                    sampleDict[key] = asample[idx]
                allSamples.append(sampleDict)
                break
            else:
                break
    with open(samplePath, 'wb') as f:
        pickle.dump(allSamples, f)


if __name__ == '__main__':
    srcDir = 'Path/to/GoPro/train'
    samplePath = 'Path/to/GoPro/gopro_train_lmdb/sample.pkl'
    numIter = 7
    genSampleTrain(srcDir, samplePath, numIter=numIter)

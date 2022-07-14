from pathlib import Path
import re
import pickle
from lib import fileTool as FT


def sortFunc(name: str):
    idx = re.search(r'(\d*).jpg', str(name)).group(1)
    return int(idx)


def extractKey(name: str):
    parent = Path(name).parent.name
    fileName = Path(name).name
    key = f'{parent}/{fileName}'
    return key


def genSampleTest(srcDir, samplePath, numIter):
    allSubDirs = FT.getSubDirs(srcDir)
    allSamples = []

    for subDir in allSubDirs:
        allFrames = FT.getAllFiles(subDir, 'jpg')
        allFrames.sort(key=sortFunc)

        for startIdx in range(0, len(allFrames) - 6, 2):
            endIdx = startIdx + numIter * 3 + 4
            if endIdx <= len(allFrames):
                sampleDict={}
                asample = allFrames[startIdx:endIdx]
                sampleDict['In'] = extractKey(asample[0])
                sampleDict['I0'] = extractKey(asample[2])
                sampleDict['I1'] = extractKey(asample[4])
                sampleDict['I2'] = extractKey(asample[6])
                sampleDict['It'] = extractKey(asample[3])

                allSamples.append(sampleDict)

    with open(samplePath, 'wb') as f:
        pickle.dump(allSamples, f)


if __name__ == '__main__':
    numIter = 1
    srcDir = 'Path/to/davis_frame'
    samplePath = 'Path/to/davis/davis_lmdb/sample.pkl'

    genSampleTest(srcDir=srcDir, samplePath=samplePath, numIter=numIter)

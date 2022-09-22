from pathlib import Path
import re
import pickle
from lib import fileTool as FT
from tqdm import tqdm


def extractKey(name: str):
    parts = Path(name).parts
    key = f'{parts[-3]}/{parts[-2]}/{parts[-1]}'
    return key


def genSampleTest(srcDir, samplePath, numIter):
    allSubDirs1 = FT.getSubDirs(srcDir)
    allSamples = []

    for subDir1 in tqdm(allSubDirs1):
        allsubDirs2 = FT.getSubDirs(subDir1)
        for subDir2 in allsubDirs2:
            allFrames = FT.getAllFiles(subDir2, 'png')
            sampleDict = {}

            for aPNG in allFrames:
                name = Path(aPNG).stem
                if name == 'im1':
                    sampleDict['In'] = extractKey(aPNG)
                if name == 'im3':
                    sampleDict['I0'] = extractKey(aPNG)
                if name == 'im5':
                    sampleDict['I1'] = extractKey(aPNG)
                if name == 'im7':
                    sampleDict['I2'] = extractKey(aPNG)
                if name == 'im4':
                    sampleDict['It'] = extractKey(aPNG)

            allSamples.append(sampleDict)

    with open(samplePath, 'wb') as f:
        pickle.dump(allSamples, f)


if __name__ == '__main__':
    numIter = 1
    srcDir = 'Path/to/vimeo_septuplet/train'  # 'Path/to/vimeo_septuplet/test'
    samplePath = 'Path/to/vimeo/train_lmdb/sample.pkl'  # 'Path/to/vimeo/test_lmdb/sample.pkl'

    genSampleTest(srcDir=srcDir, samplePath=samplePath, numIter=numIter)

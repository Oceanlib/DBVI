import pickle
import re
from pathlib import Path


def extractKey(name: str):
    parts = Path(name).parts
    key = f'{parts[-3]}/{parts[-2]}/{parts[-1]}'
    return key


def extendframe(I0: str, It: str, I1: str):
    parent = Path(I0).parent
    I0Idx = Path(I0).stem
    ItIdx = Path(It).stem
    I1Idx = Path(I1).stem
    len0 = len(I0Idx)
    gap = int(I1Idx) - int(I0Idx)
    In = str(parent / f'{int(I0Idx) - gap:0{len0}d}.png')
    I2 = str(parent / f'{int(I1Idx) + gap:0{len0}d}.png')
    return In, I0, I1, I2, It


def main(dataPath, txtPath, picklePath):
    allSamples=[]
    with open(txtPath, mode='r') as f:
        lines = f.readlines()
    for line in lines:
        sampleDict = {}
        I0, It, I1 = line.strip('\n').split(' ')
        I0, It, I1 = [extractKey(i) for i in [I0, It, I1]]
        In, I0, I1, I2, It = extendframe(I0, It, I1)

        sampleDict['In'] = In
        sampleDict['I0'] = I0
        sampleDict['It'] = It
        sampleDict['I1'] = I1
        sampleDict['I2'] = I2
        if all([(Path(dataPath) / i).is_file() for i in [In, I0, I1, I2, It]]):
            allSamples.append(sampleDict)

    with open(picklePath, 'wb') as f:
        pickle.dump(allSamples, f)


if __name__ == '__main__':
    dataPath = 'Path/to/snufilm-test'

    txtPath = 'Path/to/snufilm-test/eval_modes/test-extreme.txt'  # easy, hard, medium extreme
    picklePath = 'Path/to/snufilm/snufilm_lmdb/test-extreme.pkl'  # easy, hard, medium extreme
    main(dataPath, txtPath, picklePath)

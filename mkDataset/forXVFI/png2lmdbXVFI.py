from lib import fileTool as FT
from tqdm import tqdm
import os
import lmdb
import pickle
import cv2
import re
from pathlib import Path
from collections import OrderedDict


def extractKey(name: str):
    parts = Path(name).parts
    key = f'{parts[-2]}/{parts[-1]}'
    return key


def png2LMDB(srcDir, lmdb_path):
    allPNGs = FT.getAllFiles(srcDir, 'png')

    pbar = tqdm(total=len(allPNGs))

    isdir = os.path.isdir(lmdb_path)
    write_frequency = 20
    db = lmdb.open(lmdb_path, subdir=isdir, map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)
    txn = db.begin(write=True)
    for idx, aPNG in enumerate(allPNGs):
        key = extractKey(aPNG)

        img = cv2.imread(aPNG)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        sample = pickle.dumps(img, protocol=4)
        txn.put(u'{}'.format(key).encode('ascii'), sample)
        if idx % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)
        pbar.update(1)

    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys, protocol=4))
        txn.put(b'__len__', pickle.dumps(len(keys), protocol=4))

    print("Flushing database ...")
    db.sync()
    db.close()


def mergeData():
    srcPath = '/home/sensetime/data/VideoInterpolation/highfps/X4K1000FPS/test'
    dstPath = '/home/sensetime/data/VideoInterpolation/highfps/X4K1000FPS/mytest'
    allTypes = FT.getSubDirs(srcPath)
    total = 0
    for aType in allTypes:
        allseqs = FT.getSubDirs(aType)
        allseqs = [str(i) for i in allseqs]
        INames = OrderedDict({'0968.png': 'In.png', '0000.png': 'I0.png', '0032.png': 'I1.png', '1064.png': 'I2.png',
                              '0004.png': 'It1.png', '0008.png': 'It2.png', '0012.png': 'It3.png',
                              '0016.png': 'It4.png', '0020.png': 'It5.png', '0024.png': 'It6.png',
                              '0028.png': 'It7.png'})
        for aseq in allseqs:
            Is = [str(Path(aseq) / i) for i in INames.keys()]
            newDirName = str(Path(dstPath) / f'{Path(aType).stem}_{Path(aseq).stem}')
            FT.mkPath(newDirName)
            for I in Is:
                key = re.search('(\d){4}.png', I).group(0)
                value = INames[key]
                dstName = str(Path(newDirName) / value)
                FT.copyFile(I, dstName)
                pass

            pass


if __name__ == '__main__':
    # mergeData()
    srcDir = 'Path/to/X4K1000FPS/train'
    lmdb_path = 'Path/to/train_lmdb'
    samplePath = 'Path/to/train_lmdb/samples.pkl'
    png2LMDB(srcDir, lmdb_path)
    # checkTest()

import os

os.environ['OMP_NUM_THREADS'] = '1'
from lib import fileTool as FT
from tqdm import tqdm
import os
import lmdb
import pickle
import cv2
from pathlib import Path


def extractKey(name: str):
    parent = Path(name).parent.name
    fileName = Path(name).name
    key = f'{parent}/{fileName}'
    return key


def png2LMDB(srcDir, lmdb_path):
    allPNGs = FT.getAllFiles(srcDir, 'jpg')
    pbar = tqdm(total=len(allPNGs))

    isdir = os.path.isdir(lmdb_path)
    write_frequency = 1000
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


if __name__ == '__main__':
    srcDir = 'Path/to/davis_frame'
    lmdb_path = 'Path/to/davis/davis_lmdb'
    png2LMDB(srcDir, lmdb_path)

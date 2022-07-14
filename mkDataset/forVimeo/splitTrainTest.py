import lib.fileTool as FT
from pathlib import Path
from tqdm import tqdm

def split(srcDir, trainDir, testDir, sepTrain, sepTest):
    with open(sepTrain, 'r') as f:
        trainlist = f.read().splitlines()
    with open(sepTest, 'r') as f:
        testlist = f.read().splitlines()
    for atrain in tqdm(trainlist, leave=False):
        src = str(Path(srcDir) / atrain)
        assert Path(src).is_dir()
        dst = src.replace(srcDir, trainDir)
        FT.movFile(src, dst)
        assert Path(dst).is_dir()
    for atest in tqdm(testlist):
        src = str(Path(srcDir) / atest)
        assert Path(src).is_dir()
        dst = src.replace(srcDir, testDir)
        FT.movFile(src, dst)
        assert Path(dst).is_dir()
    pass


if __name__ == '__main__':
    srcDir = 'Path/to/vimeo_septuplet/sequences'
    trainDir = 'Path/to/vimeo_septuplet/train'
    testDir = 'Path/to/vimeo_septuplet/test'
    sepTrain = 'Path/to/vimeo_septuplet/sep_trainlist.txt'
    sepTest = 'Path/to/vimeo_septuplet/sep_testlist.txt'

    split(srcDir, trainDir, testDir, sepTrain, sepTest)

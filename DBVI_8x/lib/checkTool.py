import numpy as np
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import torch
import torch.nn.functional as F
from lib import fileTool as FT
import torch


def checkGrad(net):
    for parem in list(net.named_parameters()):
        if parem[1].grad is not None:
            print(parem[0] + ' \t shape={}, \t mean={}, \t std={}\n'.format(parem[1].shape,
                                                                            parem[1].grad.abs().mean().cpu().item(),
                                                                            parem[1].grad.abs().std().cpu().item()))


def write_video_cv2(allFrames, video_name, fps, sizes):
    out = cv2.VideoWriter(video_name, cv2.CAP_OPENCV_MJPEG, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, sizes)

    for outF in allFrames:
        # frameIn = cv2.imread(inF, cv2.IMREAD_COLOR)
        frameOut = cv2.imread(outF, cv2.IMREAD_COLOR)
        # frame = np.concatenate([frameIn, frameOut], axis=1)
        out.write(frameOut)
    out.release()


if __name__ == '__main__':
    videoPath = '/home/sensetime/data/ICCV2021/OurResults/slomoDVS34_16/Ours_S2/slomoDVS-2021_02_24_11_48_40'

    allFrames = FT.getAllFiles(videoPath, 'png')
    inFrames = [a for a in allFrames if 'EVI' not in a]
    inFrames = [[a, a, a, a] for a in inFrames]
    inFrames = [item for sublist in inFrames for item in sublist]

    videoName = '/home/sensetime/data/ICCV2021/OurResults/slomoDVS34_16/Ours_S2/slomoDVS-2021_02_24_11_48_40/couple.avi'
    write_video_cv2(inFrames, allFrames, videoName, 24, (480, 176))

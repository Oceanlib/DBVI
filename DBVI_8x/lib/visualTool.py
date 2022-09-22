import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import torch
import numpy as np
from pathlib import Path
from lib import fileTool as FT


class visFlow():
    def __init__(self, flow_uv):
        super(visFlow, self).__init__()
        self.colorWheel = self.make_colorwheel()
        self.run(flow_uv)

    def run(self, flow_uv: torch.Tensor, clip_flow=None, convert_to_bgr=False):
        flow_uv = flow_uv[0].permute([1, 2, 0]).detach().cpu().numpy()
        if clip_flow is not None:
            flow_uv = np.clip(flow_uv, 0, clip_flow)
        u = flow_uv[:, :, 0]
        v = flow_uv[:, :, 1]
        rad = np.sqrt(np.square(u) + np.square(v))
        rad_max = np.max(rad)
        epsilon = 1e-5
        u = u / (rad_max + epsilon)
        v = v / (rad_max + epsilon)
        rgb = self.flow_uv_to_colors(u, v, convert_to_bgr)
        cv2.namedWindow('flow', 0)
        cv2.imshow('flow', rgb[:, :, [2, 1, 0]])
        cv2.waitKey(0)

    def make_colorwheel(self):
        RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
        ncols = RY + YG + GC + CB + BM + MR
        colorwheel = np.zeros((ncols, 3))
        col = 0

        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
        col = col + RY
        # YG
        colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
        colorwheel[col:col + YG, 1] = 255
        col = col + YG
        # GC
        colorwheel[col:col + GC, 1] = 255
        colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
        col = col + GC
        # CB
        colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
        colorwheel[col:col + CB, 2] = 255
        col = col + CB
        # BM
        colorwheel[col:col + BM, 2] = 255
        colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
        col = col + BM
        # MR
        colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
        colorwheel[col:col + MR, 0] = 255
        return colorwheel

    def flow_uv_to_colors(self, u, v, convert_to_bgr=False):
        flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
        ncols = self.colorWheel.shape[0]
        rad = np.sqrt(np.square(u) + np.square(v))
        a = np.arctan2(-v, -u) / np.pi
        fk = (a + 1) / 2 * (ncols - 1)
        k0 = np.floor(fk).astype(np.int32)
        k1 = k0 + 1
        k1[k1 == ncols] = 0
        f = fk - k0
        for i in range(self.colorWheel.shape[1]):
            tmp = self.colorWheel[:, i]
            col0 = tmp[k0] / 255.0
            col1 = tmp[k1] / 255.0
            col = (1 - f) * col0 + f * col1
            idx = (rad <= 1)
            col[idx] = 1 - rad[idx] * (1 - col[idx])
            col[~idx] = col[~idx] * 0.75  # out of range
            # Note the 2-i => BGR instead of RGB
            ch_idx = 2 - i if convert_to_bgr else i
            flow_image[:, :, ch_idx] = np.floor(255 * col)
        return flow_image


class tensor2Video(object):
    def __init__(self, outPath, h, w, fps=24):
        super(tensor2Video, self).__init__()
        fourcc = cv2.VideoWriter.fourcc('I', '4', '2', '0')
        self.out = cv2.VideoWriter(outPath, fourcc, fps, (w, h))

    def add(self, frame: torch.Tensor):
        frame = (frame + 1.0) / 2.0
        # frame = (frame - frame.min()) / (frame.max() - frame.min())
        frame = (frame[0].permute([1, 2, 0]).cpu().numpy() * 255).astype(np.uint8)
        self.out.write(frame)

    def release(self):
        self.out.release()


def makeGrid(batchImg: torch.Tensor, shape=(2, 1)):
    N, C, H, W = batchImg.shape
    batchImg = batchImg.permute([0, 2, 3, 1])
    batchImg = batchImg.detach().cpu().numpy()
    nh = shape[0]
    nw = shape[1]
    batchImg = batchImg.reshape((nh, nw, H, W, C)).swapaxes(1, 2).reshape(nh * H, nw * W, C)
    return batchImg


def visImg(batchImg: torch.Tensor, shape=(1, 1), wait=0, name='visImg'):
    """
    :param img: tensor(N,3,H,W) or None
    :return: None
    """
    N, C, H, W = batchImg.shape
    assert all([C == 3, N == shape[0] * shape[1]])
    # batchImg = ((batchImg - batchImg.min()) / (batchImg.max() - batchImg.min()) * 255.0).byte()
    batchImg = ((batchImg.float() + 1) / 2.0 * 255).clamp(0, 255).byte()
    batchImgViz = makeGrid(batchImg, shape)
    cv2.namedWindow(name, 0)
    cv2.imshow(name, batchImgViz[:, :, ::-1])
    cv2.waitKey(wait)


def saveImg(x: torch.Tensor, srcName: str, dstDir: str, isInter=False):
    if isInter:
        dstName = str(Path(dstDir) / srcName.replace('.png', '_inter.png'))
    else:
        dstName = str(Path(dstDir) / srcName)

    if Path(dstName).is_file():
        return False
    if not Path(Path(dstName).parent).is_dir():
        FT.mkPath(Path(dstName).parent)

    xRGB = (x.clamp(-1, 1) + 1.0) / 2.0
    xRGB = (xRGB[0].permute([1, 2, 0]).detach().cpu() * 255).byte().numpy()
    xBGR = cv2.cvtColor(xRGB, cv2.COLOR_RGB2BGR)
    cv2.imwrite(dstName, xBGR)
    return True


def saveTensor(x: torch.Tensor, srcName: str, dstDir: str, isInter=False):
    if isInter:
        dstName = str(Path(dstDir) / srcName.replace('.png', '_inter.pth'))
    else:
        dstName = str(Path(dstDir) / srcName.replace('.png', '.pth'))

    if Path(dstName).is_file():
        return False
    if not Path(Path(dstName).parent).is_dir():
        FT.mkPath(Path(dstName).parent)

    # xRGB = (x + 1.0) / 2.0
    xRGB = x.detach().cpu()
    torch.save(xRGB, dstName)
    return True



def visImg(batchImg: torch.Tensor, shape=(1, 1), wait=0, name='visImg'):
    """
    :param img: tensor(N,3,H,W) or None
    :return: None
    """
    N, C, H, W = batchImg.shape
    assert all([C == 3, N == shape[0] * shape[1]])
    # batchImg = ((batchImg - batchImg.min()) / (batchImg.max() - batchImg.min()) * 255.0).byte()
    batchImg = ((batchImg.float() + 1) / 2.0 * 255).clamp(0, 255).byte()[:, [2, 1, 0], :, :]  # rgb2bgr
    batchImgViz = makeGrid(batchImg, shape)
    cv2.namedWindow(name, 0)
    cv2.imshow(name, batchImgViz)
    cv2.waitKey(wait)


def saveImg(x: torch.Tensor, outpath, isrgb=True):
    if Path(outpath).is_file():
        return None
    rmax, rmin = x.max(), x.min()
    if rmin < -0.5:
        x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    if isrgb:
        x = x[:, [2, 1, 0], :, :]
    xNpy = (x.squeeze(0).permute([1, 2, 0]) * 255).byte().detach().cpu().numpy()
    FT.mkPath(Path(outpath).parent)

    cv2.imwrite(str(outpath), xNpy)
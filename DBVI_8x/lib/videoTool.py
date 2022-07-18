import os
import lib.fileTool as FT

ffmpegPath = '/usr/bin/ffmpeg'
videoName = '/home/sensetime/data/VideoInterpolation/highfps/gopro_yzy/bbb.MP4'


def video2Frame(vPath: str, fdir: str, H: int = None, W: int = None):
    FT.mkPath(fdir)
    if H is None or W is None:
        os.system('{} -y -i {} -vsync 0 -qscale:v 2 {}/%04d.png'.format(ffmpegPath, vPath, fdir))
    else:
        os.system('{} -y -i {} -vf scale={}:{} -vsync 0 -qscale:v 2 {}/%04d.jpg'.format(ffmpegPath, vPath, W, H, fdir))


def frame2Video(fdir: str, vPath: str, fps: int, H: int = None, W: int = None, ):
    if H is None or W is None:
        # os.system('{} -y -r {} -f image2 -i {}/%*.png -vcodec libx264 -crf 18 -pix_fmt yuv420p {}'
        #           .format(ffmpegPath, fps, fdir, vPath))

        os.system('{} -y -r {} -f image2 -i {}/%4d.png -vcodec libx264 -crf 18 -pix_fmt yuv420p {}'
                  .format(ffmpegPath, fps, fdir, vPath))
    else:
        os.system('{} -y -r {} -f image2 -s {}x{} -i {}/%*.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {}'
                  .format(ffmpegPath, fps, W, H, fdir, vPath))


def slomo(vPath: str, dstPath: str, fps):
    os.system(
        '{} -y -r {} -i {}  -strict -2 -vcodec libx264 -c:a aac -crf 18 {}'.format(ffmpegPath, fps, vPath, dstPath))


def downFPS(vPath: str, dstPath: str, fps):
    os.system(
        '{} -i {}  -strict -2 -r {} {}'.format(ffmpegPath, vPath, fps, dstPath))


def downSample(vPath: str, dstPath: str, H, W):
    os.system(
        '{} -i {}  -strict -2 -s {}x{}  {}'.format(ffmpegPath, vPath, H, W, dstPath))


if __name__ == '__main__':
    framePath = '/data/2021_12_23/dstframes'
    # framePath = '/home/sensetime/data/VideoInterpolation/highfps/gopro_yzy/output'
    video = '/data/2021_12_23/02.mp4'
    # video2Frame(video, framePath)

    # video = '/home/sensetime/data/VideoInterpolation/highfps/goPro_240fps/train/GOPR0372_07_00/out.mp4'
    # framePath = '/home/sensetime/data/VideoInterpolation/highfps/goPro_240fps/train/GOPR0372_07_00'
    frame2Video(framePath, video, 30)

    # vPath = '/media/sensetime/Elements/0721   /0716_video/1.avi'
    # dstPath = '/media/sensetime/Elements/0721/0716_video/1_.mp4'
    # downFPS(vPath, dstPath, 8)

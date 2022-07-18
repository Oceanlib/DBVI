from __future__ import division
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import math
import torch
import random
import numbers
import numpy as np
import collections

INTER_MODE = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'BICUBIC': cv2.INTER_CUBIC}
PAD_MOD = {'constant': cv2.BORDER_CONSTANT,
           'edge': cv2.BORDER_REPLICATE,
           'reflect': cv2.BORDER_DEFAULT,
           'symmetric': cv2.BORDER_REFLECT
           }


class Resize(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size=None, scale=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.scale = scale

    def method(self, img):

        h, w, _ = img.shape
        if self.scale is not None:
            img = cv2.resize(src=img, dsize=(0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        elif any([w != self.size[0], h != self.size[1]]):
            img = cv2.resize(src=img, dsize=self.size, interpolation=cv2.INTER_CUBIC)

        return img

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        return [self.method(img) for img in imgs]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def method(self, img):

        h, w, _ = img.shape
        th, tw = self.size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        return img[i:i + th, j:j + tw, :]

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        return [self.method(img) for img in imgs]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomCrop(object):
    """Vertically flip the given CV Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def get_params(self, imgs):
        h, w, _ = imgs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return i, j, th, tw

    def method(self, img, i, j, h, w):
        return img[i:i + h, j:j + w, :]

    def __call__(self, imgs):
        i, j, h, w = self.get_params(imgs)

        return [self.method(img, i, j, h, w) for img in imgs]


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
           transforms.Compose([
             transforms.CenterCrop(10),
             transforms.ToTensor(),
         ])
    """
    __slots__ = ['transforms']

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    __slots__ = []

    def method(self, pic):
        if _is_numpy_image(pic):
            if len(pic.shape) == 2:
                pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            if isinstance(img, torch.ByteTensor) or img.max() > 1:
                return img.float().div(255.0).mul(2.0).add(-1.0)
                # return img.float().div(255.0)
            else:
                return img
        elif _is_tensor_image(pic):
            return pic

        else:
            try:
                return self.method(np.array(pic))
            except Exception:
                raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))

    def __call__(self, pic):
        """
        Args:
            pic (numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if not isinstance(pic, list):
            pic = [pic]
        return [self.method(i) for i in pic]

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomHVerticalFlip(object):
    """Vertically flip the given CV Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def method(self, img: np.ndarray):
        if not _is_numpy_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        if img.shape[2] == 2:
            img = np.concatenate(img, img[:, :, 0:1], axis=2)
        return cv2.flip(img[:, :, 0:img.shape[2]], 0)

    def __call__(self, img):
        """
        Args:
            img (CV Image): Image to be flipped.

        Returns:
            CV Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return [self.method(i) for i in img]
        return img


class RandomHorizontalFlip(object):
    """Vertically flip the given CV Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def method(self, img):
        if not _is_numpy_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
        if img.shape[2] == 2:
            img = np.concatenate(img, img[:, :, 0:1], axis=2)
        return cv2.flip(img[:, :, 0:img.shape[2]], 1)

    def __call__(self, img):
        """
        Args:
            img (CV Image): Image to be flipped.

        Returns:
            CV Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return [self.method(i) for i in img]
        return img


class AdjustBrightness(object):
    __slots__ = ['brightness_factor']

    def __init__(self, brightness_factor):
        self.brightness_factor = brightness_factor

    def method(self, img):
        if not _is_numpy_image(img):
            raise TypeError('img should be CV Image. Got {}'.format(type(img)))

        im = img.astype(np.float32) * self.brightness_factor
        im = im.clip(min=0, max=255)
        return im.astype(img.dtype)

    def __call__(self, img):
        return [self.method(i) for i in img]


class AdjustContrast(object):
    __slots__ = ['contrast_factor']

    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def method(self, img):
        if not _is_numpy_image(img):
            raise TypeError('img should be CV Image. Got {}'.format(type(img)))
        im = img.astype(np.float32)
        mean = round(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).mean())
        im = (1 - self.contrast_factor) * mean + self.contrast_factor * im
        im = im.clip(min=0, max=255)
        return im.astype(img.dtype)

    def __call__(self, img):
        return [self.method(i) for i in img]


class AdjustSaturation(object):
    __slots__ = ['saturation_factor']

    def __init__(self, saturation_factor):
        self.saturation_factor = saturation_factor

    def method(self, img):
        if not _is_numpy_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        im = img.astype(np.float32)
        degenerate = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        im = (1 - self.saturation_factor) * degenerate + self.saturation_factor * im
        im = im.clip(min=0, max=255)
        return im.astype(img.dtype)

    def __call__(self, img):
        return [self.method(i) for i in img]


class AdjustHue(object):
    __slots__ = ['hue_factor']

    def __init__(self, hue_factor):
        self.hue_factor = hue_factor

    def method(self, img):
        if not (-0.5 <= self.hue_factor <= 0.5):
            raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(self.hue_factor))

        if not _is_numpy_image(img):
            raise TypeError('img should be CV Image. Got {}'.format(type(img)))

        im = img.astype(np.uint8)
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV_FULL)
        hsv[..., 0] += np.uint8(self.hue_factor * 255)

        im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
        return im.astype(img.dtype)

    def __call__(self, img):
        return [self.method(i) for i in img]


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    __slots__ = ['brightness', 'contrast', 'saturation', 'hue']

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(AdjustBrightness(brightness_factor))

        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(AdjustContrast(contrast_factor))

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(AdjustSaturation(saturation_factor))

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
            transforms.append(AdjustHue(hue_factor))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Color jittered image.
        """

        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)


if __name__ == '__main__':
    for i in range(10000):
        image_path = './dataloader/timg.jpg'

        cvimage = cv2.imread(image_path, cv2.IMREAD_COLOR)
        cvimage_gray = cv2.cvtColor(cvimage, cv2.COLOR_RGB2GRAY)
        imglist = [cvimage, cvimage_gray]

        methods = Compose(method)
        imglist = methods(imglist)
        # sub = imglist[0] - imglist[1]
        # cv2.namedWindow('1', 0)
        cv2.imshow('1', imglist[0])
        cv2.imshow('2', imglist[1])
        cv2.waitKey(0)
        # imshow(imglist + [sub], ('1', '2', '3'))

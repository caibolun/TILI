#!/usr/bin/env python
# coding=utf-8
'''
@Author: ArlenCai
@Date: 2020-02-04 14:33:22
@LastEditTime : 2020-02-09 21:28:14
'''
import sys
import os
import collections
import numbers
import warnings
import math
import random
import numpy as np
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import torch
from torch.nn.functional import interpolate

from . import functional as F

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

__all__ = ["Compose", "ImageDecoder", "Resize", "CenterCrop", "Pad", "Normalize", "ResizedCrop",
"HorizontalFlip", "VerticalFlip", "CvtColor"]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ImageDecoder(object):
    def __init__(self, max_size=0, use_tensor=True):
        self.max_size = max_size**2
        self.use_tensor = use_tensor

    def __call__(self, filepath):
        if accimage is not None:
            sample = accimage.Image(filepath)
        else:
            with open(filepath, 'rb') as f:
                sample = Image.open(f)
                sample = sample.convert('RGB')
        if self.use_tensor:
            if self.max_size !=0 and sample.width * sample.height > self.max_size:
                ratio = math.sqrt(self.max_size/(sample.width * sample.height))
                sample = sample.resize((int(ratio*sample.width), int(ratio*sample.height)), Image.NEAREST)
            sample = F.img2tensor(sample)
            if torch.cuda.is_available():
                sample = sample.cuda()
        return sample


class Resize(object):
    def __init__(self, size, interpolation='bilinear'):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return F.resize(img, self.size, self.interpolation)


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        return F.center_crop(img, self.size)


class Pad(object):
    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'reflect', 'replicate', 'circular']
        if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        return F.pad(img, self.padding, self.fill, self.padding_mode)
        

class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        return F.normalize(tensor, self.mean, self.std, self.inplace)


class ResizedCrop(object):
    def __init__(self, size, interpolation='bilinear'):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return F.resized_crop(img, self.size, self.interpolation)


class HorizontalFlip(object):
    def __call__(self, img):
        return F.hflip(img)


class VerticalFlip(object):
    def __call__(self, img):
        return F.vflip(img)

class CvtColor(object):
    def __init__(self, code):
        assert code in ['RGB2BGR', 'BGR2RGB', 'BGR2GRAY', 'RGB2GRAY']
        self.code = code
    def __call__(self, img):
        return F.cvtColor(img, self.code)
        




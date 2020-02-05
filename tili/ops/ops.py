#!/usr/bin/env python
# coding=utf-8
'''
@Author: ArlenCai
@Date: 2020-02-04 14:33:22
@LastEditTime : 2020-02-05 00:46:17
'''
import os
import math
import torch
from torch.utils import data
from torch.nn.functional import interpolate
#import accimage
from . import functional as F
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
class FileReader(object):
    def __init__(self, file_root, extensions=IMG_EXTENSIONS):
        self.file_root = os.path.expanduser(file_root)
        self.extensions = extensions
    def __call__(self):
        files = []
        for root, _, fnames in sorted(os.walk(self.file_root, followlinks=True)):
            for fname in sorted(fnames):
                filename = os.path.join(root, fname)
                if filename.lower().endswith(self.extensions):
                    files.append(filename)
        return files
'''
class ImageDecoder(data.Dataset): 
    def __init__(self, data, max_size=256, cuda=False):
        if isinstance(data, str):
            self.samples = [data]
        elif isinstance(data, list):
            if len(data) == 0:
                raise(RuntimeError("Found 0 filepaths"))
            self.samples = data
        else:
            raise(RuntimeError("The type of data is error."))
        self.max_size = max_size**2
        self.cuda = cuda

    def __getitem__(self, index):
        sample = Image.open(self.samples[index])
        if sample.width * sample.height > self.max_size:
            ratio = math.sqrt(self.max_size/(sample.width * sample.height))
            sample = sample.resize((int(ratio*sample.width), int(ratio*sample.height)), Image.NEAREST)
        sample = F.img2tensor(sample)
        if self.cuda:
            sample = sample.cuda()
        return sample

    def __len__(self):
        return len(self.samples)
'''

class ImageDecoder(object):
    def __init__(self, max_size=256, cuda=False):
        self.max_size = max_size**2
        self.cuda = cuda

    def __call__(self, filepath):
        sample = Image.open(filepath)
        if sample.width * sample.height > self.max_size:
            ratio = math.sqrt(self.max_size/(sample.width * sample.height))
            sample = sample.resize((int(ratio*sample.width), int(ratio*sample.height)), Image.NEAREST)
        sample = F.img2tensor(sample)
        if self.cuda:
            sample = sample.cuda()
        return sample

class Resize(object):
    def __init__(self, size, interpolation='bilinear'):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return F.resize(img, self.size, self.interpolation)

    
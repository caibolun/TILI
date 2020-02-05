#!/usr/bin/env python
# coding=utf-8
'''
@Author: ArlenCai
@Date: 2020-02-04 17:27:50
@LastEditTime : 2020-02-05 00:38:27
'''
import torch
import numpy as np
from torch.nn import functional as F
try:
    import accimage
except ImportError:
    accimage = None

def img2tensor(pic):
    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    img.unsqueeze_(0)
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img

def resize(img, size, interpolation='bilinear', align_corners=True):
    if isinstance(size, int):
        n, c, w, h = img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            img = F.interpolate(img, (ow, oh), mode=interpolation, align_corners=align_corners)
            return img
        else:
            oh = size
            ow = int(size * w / h)
            img = F.interpolate(img, (ow, oh), mode=interpolation, align_corners=align_corners)
            return img
    else:
        img = F.interpolate(img, size[::-1], mode=interpolation, align_corners=align_corners)
        return img
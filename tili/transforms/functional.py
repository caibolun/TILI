#!/usr/bin/env python
# coding=utf-8
'''
@Author: ArlenCai
@Date: 2020-02-04 17:27:50
@LastEditTime : 2020-02-09 18:29:15
'''
import os
import math
import random
import numbers
import numpy as np
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import torch
from torch.nn import functional as F

def img2tensor(pic):
    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        img = torch.from_numpy(nppic)
        img.unsqueeze_(0)
        return img

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
        img = img.float().div(255)
    if torch.cuda.is_available():
        img = img.cuda()
    return img

INTER_MODE = {
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC
}

def resize_tensor(img, size, interpolation='bilinear'):
    if isinstance(size, int):
        n, c, h, w = img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return F.interpolate(img, (oh, ow), mode=interpolation, align_corners=True)
    else:
        return F.interpolate(img, size, mode=interpolation, align_corners=True)

def resize_image(img, size, interpolation='bilinear'):
    interpolation = INTER_MODE[interpolation]
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)

def resize(img, size, interpolation='bilinear'):
    if isinstance(img, torch.Tensor):
        return resize_tensor(img, size, interpolation)
    else:
        img = resize_image(img, size, interpolation)
        return img2tensor(img)

def pad(img, padding, fill=0, padding_mode='constant'):
    assert padding_mode in ['constant', 'reflect', 'replicate', 'circular'], \
        'Padding mode should be either constant, reflect, replicate or circular'
    return F.pad(img, padding, padding_mode, fill)

def crop(img, top, left, height, width):
    return img[:, :, top:top+height, left:left+width]

def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    n, c, image_height, image_width = img.shape
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return crop(img, crop_top, crop_left, crop_height, crop_width)

def resized_crop_tensor(img, size, interpolation='bilinear'):
    n, c, h, w = img.shape
    if isinstance(size, int):
        if (w <= h and w == size) or (h <= w and h == size):
            return center_crop(img, size)
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        img = img.resize(img, (oh, ow), mode=interpolation, align_corners=True)
        return center_crop(img, size)
    else:
        scale_h = size[0]/h
        scale_w = size[1]/w
        if scale_h == scale_w:
            oh = size[0]
            ow = size[1]
        elif scale_h > scale_w:
            oh = size[0]
            ow = int(scale_h * w)
        else:
            ow = size[1]
            oh = int(scale_w * h)
        img = F.interpolate(img, (oh, ow), mode=interpolation, align_corners=True)
        return center_crop(img, size)

def resized_crop_image(img, size, interpolation='bilinear'):
    interpolation = INTER_MODE[interpolation]
    w, h = img.size
    if isinstance(size, int):
        if (w <= h and w == size) or (h <= w and h == size):
            return center_crop(img, size)
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        img = img.resize((ow, oh), interpolation)
        img_tensor = img2tensor(img)
        return center_crop(img_tensor, size)
    else:
        scale_h = size[0]/h
        scale_w = size[1]/w
        if scale_h == scale_w:
            oh = size[0]
            ow = size[1]
        elif scale_h > scale_w:
            oh = size[0]
            ow = int(scale_h * w)
        else:
            ow = size[1]
            oh = int(scale_w * h)
        img = img.resize((ow, oh), interpolation)
        img_tensor = img2tensor(img)
        return center_crop(img_tensor, size)

def resized_crop(img, size, interpolation='bilinear'):
    if isinstance(img, torch.Tensor):
        return resized_crop_tensor(img, size, interpolation)
    else:
        return resized_crop_image(img, size, interpolation)

def normalize(tensor, mean, std, inplace=False):
    if not inplace:
        tensor = tensor.clone()
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor

def hflip(img):
    return img.flip(3)

def vflip(img):
    return img.flip(2)

def adjust_brightness(img, brightness_factor):
    return img.mul_(brightness_factor)

def adjust_contrast(img, contrast_factor):
    return img.add_(contrast_factor)

def cvtColor(img, code="RGB2BGR"):
    if code == "RGB2BGR":
        return img[:, [2, 1, 0],:,:]
    if code == "BGR2RGB":
        return img[:, [2, 1, 0],:,:]
    if code == "RGB2GRAY":
        weight = torch.as_tensor([0.2989, 0.5870, 0.1140], dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
        return F.conv2d(img, weight)
    if code == "BGR2GRAY":
        weight = torch.as_tensor([0.1140, 0.5870, 0.2989], dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
        return F.conv2d(img, weight[None, :, None, None])
    


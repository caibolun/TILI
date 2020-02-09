#!/usr/bin/env python
# coding=utf-8
'''
@Author: ArlenCai
@Date: 2020-02-07 16:34:00
@LastEditTime : 2020-02-09 11:55:58
'''
import time
import torch
from torch import nn
import torchvision
import os
from torchvision.datasets.folder import pil_loader, accimage_loader
from torchvision.transforms import functional as torch_fun
from tili.io import img_reader
from tili.transforms import functional as tili_fun
from tqdm import tqdm
try:
    import accimage
except:
    accimage = None

def slow_process(filename):
    if accimage is not None:
        img = accimage_loader(filename)
    else:
        img = pil_loader(filename)
    img = torch_fun.resize(img, (512, 512))
    #img = torch_fun.vflip(img)
    #img = torch_fun.hflip(img)
    img_tensor = torch_fun.to_tensor(img)
    img_tensor = torch_fun.normalize(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_tensor.unsqueeze_(0)
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    return img_tensor

def fast_process(filename):
    if accimage is not None:
        img = accimage_loader(filename)
    else:
        img = pil_loader(filename)
    img_tensor = tili_fun.img2tensor(img)
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    img_tensor = tili_fun.resize(img_tensor, (512, 512))
    img_tensor = tili_fun.normalize(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_tensor = tili_fun.vflip(img_tensor)
    img_tensor = tili_fun.hflip(img_tensor)
    return img_tensor
    

if __name__ == "__main__":
    filenames = img_reader('./imagenet_1k/')
    filename = './imagenet_1k/ILSVRC2012_val_00000114.JPEG'
    warm_num = 20
    test_num = 1000

    print(slow_process(filename).shape)
    for i in range(warm_num): img_tensor = slow_process(filename)
    start_time = time.time()
    for x in filenames[0:test_num]: img_tensor = slow_process(x)
    #for i in range(test_num): slow_process(filename)
    print("SLOW: %f s"%(time.time()-start_time))


    print(fast_process(filename).shape)
    for i in range(warm_num): img_tensor = fast_process(filename)
    start_time = time.time()
    for x in filenames[0:test_num]: img_tensor = fast_process(x)
    #for i in range(test_num): fast_process(filename, process)
    print("FAST: %f s"%(time.time()-start_time))
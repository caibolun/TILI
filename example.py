#!/usr/bin/env python
# coding=utf-8
'''
@Author: ArlenCai
@Date: 2020-02-04 17:53:40
@LastEditTime : 2020-02-05 01:02:27
'''
import torch
from tili import ops
if __name__ == "__main__":
    ops.Compose([
        ops.ImageDecoder(),
        transforms.RandomResizedCrop(args.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    ops.FileReader('./data')
    reader = FileReader('./data')
    files = reader()
    decode = ImageDecoder()
    resize = Resize(128)
    img = decode(files[0])
    img = resize(img)
    print(img.shape)
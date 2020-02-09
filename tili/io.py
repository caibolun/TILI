#!/usr/bin/env python
# coding=utf-8
'''
@Author: ArlenCai
@Date: 2020-02-06 17:19:52
@LastEditTime : 2020-02-06 18:19:50
'''
import os

__all__ = ["file_reader", "img_reader"]

def file_reader(file_root, extensions):
    file_root = os.path.expanduser(file_root)
    samples = []
    for root, _, fnames in sorted(os.walk(file_root, followlinks=True)):
        for fname in sorted(fnames):
            filename = os.path.join(root, fname)
            if filename.lower().endswith(extensions):
                samples.append(filename)
    return samples

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
def img_reader(file_root):
    return file_reader(file_root, IMG_EXTENSIONS)

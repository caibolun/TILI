#!/usr/bin/env python
# coding=utf-8
'''
@Author: ArlenCai
@Date: 2020-02-04 14:33:22
@LastEditTime : 2020-02-04 22:50:43
'''
import torch
import torchvision
from torchvision.transforms import functional as F
#import accimage
from  PIL import Image
from torch.utils import data

class Pipline(object):
    def __init__(self, batch_size=1, num_threads=1, ):
        self._batch_size = batch_size
        self._num_threads = num_threads
        
        dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_threads)

    @property
    def num_threads(self):
        """Number of CPU threads used by the pipeline."""
        return self._num_threads
    
    def __call__(self, inputs):
'''

if __name__ == "__main__":
    decoder = ImageDecoder('../data/lena.jpg')
    img = decoder.__getitem__(0)
    print(img.shape)

        
        

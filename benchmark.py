#!/usr/bin/env python
# coding=utf-8
'''
@Author: ArlenCai
@Date: 2020-02-04 17:53:40
@LastEditTime : 2020-02-09 23:26:33
'''
import multiprocessing as mp
import time
import torch

from torch.utils import data
import torchvision
from torchvision.datasets.folder import pil_loader

import tili
from tili.pipeline import Pipline, ImagePipline
from tili.io import img_reader

class ImageDataset(data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        sample = pil_loader(self.samples[index])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.samples)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    filelist = img_reader("./imagenet_1k")

    torch_trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 512)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = ImageDataset(filelist, torch_trans)
    dataloader = data.DataLoader(dataset, batch_size=8, num_workers=1)
    start_time = time.time()
    for batch_idx, sample in enumerate(dataloader):
        sample = sample.cuda()
    elapsed_time = time.time() - start_time
    print("[Torch] Batch Used Time: %f s"%elapsed_time)
    start_time = time.time()
    for x in filelist:
        sample = pil_loader(x)
        sample = torch_trans(sample)
        sample.unsqueeze_(0)
        sample = sample.cuda()
    elapsed_time = time.time() - start_time
    print("[Torch] Image Used Time: %f s"%elapsed_time)
    print(type(sample), sample.device, sample.shape)

    print("\n")

    tili_trans = tili.transforms.Compose([
        tili.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    pipe = ImagePipline(tili_trans, (512, 512), batch_size=8, num_workers=1, keep_ratio=False, cuda_resize=True)
    start_time = time.time()
    for batch_idx, sample in enumerate(pipe(filelist)):
        pass
    elapsed_time = time.time() - start_time
    print("[TILI] Batch Used Time: %f s"%elapsed_time)
    start_time = time.time()
    for x in filelist:
        sample = pipe(x)
    elapsed_time = time.time() - start_time
    print("[TILI] Image Used Time: %f s"%elapsed_time)
    print(type(sample), sample.device, sample.shape)

    



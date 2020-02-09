#!/usr/bin/env python
# coding=utf-8
'''
@Author: ArlenCai
@Date: 2020-02-04 14:33:22
@LastEditTime : 2020-02-09 12:19:47
'''
import os
import torch
from torch.utils import data
from prefetch_generator import BackgroundGenerator
from . import transforms

__all__ = ["Pipline", "ImagePipline"]


class DataLoaderX(data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class _ListDataset(data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if isinstance(sample, torch.Tensor) and sample.dim()==4 and sample.size(0)==1:
            sample.squeeze_(0)
        return sample

    def __len__(self):
        return len(self.samples)


class _BatchTransform(object):
    def __init__(self, transform=None):
        self.transform = transform
    def __call__(self, batch):
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
            batch_tensor = torch.stack(batch, 0, out=out)
            if torch.cuda.is_available() and not batch_tensor.is_cuda:
                batch_tensor = batch_tensor.cuda()
            if self.transform is not None:
                return self.transform(batch_tensor)
            return batch_tensor
        else:
            return batch

class _ListIter(object):
    def __init__(self, inputs, decoder=None, transform=None):
        self.inputs = inputs
        self.decoder = decoder
        self.transform = transform
        self.len = len(inputs)
        self.idx = 0

    def __next__(self):
        if self.idx < self.len:
            sample = self.inputs[self.idx]
            self.idx += 1
            if self.decoder is None:
                return sample
            elem = self.decoder(sample)
            if torch.cuda.is_available() and not elem.is_cuda:
                elem = elem.cuda()
            if self.transform is None:
                return elem
            return self.transform(elem)
        else:
            raise StopIteration

    def __iter__(self):
        self.idx = 0
        return self

    def __len__(self):
        return self.len

class Pipline(object):
    def __init__(self, decoder=None, transform=None, batch_size=1, num_workers=1):
        self.decoder = decoder
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __call__(self, inputs):
        if isinstance(inputs, list):
            if self.batch_size>1:
                dataset = _ListDataset(inputs, self.decoder)
                collate_fn = _BatchTransform(self.transform)
                dataloader = DataLoaderX(dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate_fn)
                return dataloader
            else:
                return _ListIter(inputs, self.decoder, self.transform)
        else:
            if self.decoder is None:
                return inputs
            elem = self.decoder(inputs)
            if torch.cuda.is_available() and not elem.is_cuda:
                elem = elem.cuda()
            if self.transform is None:
                return elem
            return self.transform(elem)


class ImagePipline(Pipline):
    def __init__(self, transform=None, size=None, batch_size=1, num_workers=1, keep_ratio=True, max_size=0, use_tensor=True):
        compose = [transforms.ImageDecoder(max_size, use_tensor)]
        if size is not None:
            if keep_ratio:
                compose.append(transforms.ResizedCrop(size))
            else:
                compose.append(transforms.Resize(size))
        else:
            batch_size = 1
            num_workers = 1
        decoder = transforms.Compose(compose)
        super().__init__(decoder=decoder, transform=transform, batch_size=batch_size, num_workers=num_workers)

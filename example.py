#!/usr/bin/env python
# coding=utf-8
'''
@Author: ArlenCai
@Date: 2020-02-04 17:53:40
@LastEditTime : 2020-02-11 17:10:13
'''
from labels import classname
import numpy as np
import torch
import cv2
import tili
from tili import transforms
from tili.pipeline import ImagePipline
from torchvision.models import shufflenet_v2_x0_5

if __name__ == "__main__":
    # model forward
    trans = transforms.Compose([
        transforms.ResizedCrop((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = shufflenet_v2_x0_5(pretrained=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    pipe = ImagePipline(trans)
    img = pipe('cat.jpg')
    print(type(img), img.device, img.shape)
    with torch.no_grad():
        y = model(img)
    print("Predict:", classname[y.argmax().item()])

    # other transforms
    trans = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ResizedCrop((256, 256)),
        transforms.CenterCrop(224),
        transforms.Pad((16, 16, 16, 16), padding_mode='reflect'),
        transforms.HorizontalFlip(),
        transforms.VerticalFlip(),
        transforms.CvtColor('RGB2BGR'),
    ])
    pipe = ImagePipline(trans)
    #img = pipe('cat.jpg')
    with open('cat.jpg', 'rb') as fp:
        img = pipe(fp)
    npimg = img.squeeze_(0).data.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    cv2.imshow('img', np.uint8(npimg*255))
    cv2.waitKey()


    
    
    
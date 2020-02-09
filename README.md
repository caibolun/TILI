<!--
 * @Author: ArlenCai
 * @Date: 2020-02-05 10:46:02
 * @LastEditTime : 2020-02-09 22:26:12
 -->
# TILI: Turbo Image Loading Library
## Introduction
Turbo Image Loading Library (TILI) is a collection of highly optimized library for Pytorch to accelerate the pre-processing of the image data for deep learning applications.
## Highlights
+ Full accelerated pipeline from reading the disk to getting ready for inference
+ Flexibility through configurable pipeline and custom operators
+ Accelated by libjpeg-turbo (accimage), (prefetch_generator) and pytorch (CUDA)

## Examples

```
from labels import classname
import numpy as np
import torch
import tili
from tili import transforms
from tili.pipeline import ImagePipline
from torchvision.models import mobilenet_v2
trans = transforms.Compose([
        transforms.ResizedCrop((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = mobilenet_v2(pretrained=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    pipe = ImagePipline(trans)
    img = pipe('cat.jpg')
    print(type(img), img.device, img.shape)
    with torch.no_grad():
        y = model(img)
    print("Predict:", classname[y.argmax().item()])
```

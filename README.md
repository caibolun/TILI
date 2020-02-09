<!--
 * @Author: ArlenCai
 * @Date: 2020-02-05 10:46:02
 * @LastEditTime : 2020-02-09 23:24:38
 -->
# TILI: Turbo Image Loading Library
## Introduction
Turbo Image Loading Library (TILI) is a collection of highly optimized library for Pytorch to accelerate the pre-processing of the image data for deep learning applications.
## Highlights
+ Full accelerated pipeline from reading the disk to getting ready for inference
+ Flexibility through configurable pipeline and custom operators
+ Accelated by libjpeg-turbo ([accimage](https://github.com/pytorch/accimage)), BackgroundGenerator ([prefetch_generator](https://pypi.org/project/prefetch_generator/)) and [pytorch](https://pytorch.org/) with CUDA.

## Benchmark
The benchmark run on 1K ImageNet test samples ([imagenet_1k.zip](https://download.pytorch.org/tutorial/hymenoptera_data.zip)) by Tesla P4 GPU with the simplest transforms (ImageDecoder, Resize, Normalize).
Image Size | 256x256 | 512x512 |
---|---|---|
torchvision (baseline) | 9.7818 ms/p | 12.5466 ms/p |
+ libjpeg-turbo | 4.3591 ms/p | 5.6756 ms/p |
w/o CUDA resize | **3.7838 ms/p** | 4.8411 ms/p |
w/ CUDA resize | 4.0056 ms/p | **4.0331 ms/p** |


## Usage
### Dependencies
```
$ conda install -c conda-forge accimage
$ pip install prefetch_generator
```
### Examples
1. A simple example to use TILI for inference. 
```
from labels import classname
import torch
import tili
from tili import transforms
from tili.pipeline import ImagePipline
from torchvision.models import mobilenet_v2
model = mobilenet_v2(pretrained=True)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()

trans = transforms.Compose([
        transforms.ResizedCrop((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
pipe = ImagePipline(trans)
img = pipe('cat.jpg')
with torch.no_grad():
    y = model(img)
print("Predict:", classname[y.argmax().item()])
```
2. A simple example to use the other TILI transforms.
```
import numpy as np
import torch
import tili
from tili import transforms
from tili.pipeline import ImagePipline
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
img = pipe('cat.jpg')
npimg = img.squeeze_(0).data.numpy()
npimg = np.transpose(npimg, (1, 2, 0))
cv2.imshow('img', np.uint8(npimg*255))
cv2.waitKey()
```
## Contact
Please contact ArlenCai (arlencai@tencent.com)
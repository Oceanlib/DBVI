# Official code for DBVI(ECCV2022)
[[Paper]](https://www.ecva.net/papers.php),  [[Video]](https://youtu.be/8KvFwN1_3DY),  [[Presentation]](https://youtu.be/2quo-k0PcQ4)

## 1. Requirements

1) cuda 9.0, cudnn7.6.5

2) python 3.6.9

3) pytorch 1.8.1

4) numpy 1.17.2

5) cupy-90

6) tqdm

7) gcc 5.4.0

8) cmake 3.16.0

9) opencv_contrib_python

10) [Apex](https://github.com/NVIDIA/apex) 

11) For distributed training with multi-gpus on cluster: slurm 15.08.11


## 2. How to use 
[For 8x interpolation](https://github.com/Oceanlib/DBVI/tree/main/DBVI_8x) 

[For 2x interpolation](https://github.com/Oceanlib/DBVI/tree/main/DBVI_2x)

## 3. Citation 
```
@InProceedings{DBVI_ECCV_2022,
    author    = {Yu, Zhiyang and Zhang, Yu and Xiang, Xujie and Zou, Dongqing and Chen, Xijun and Jimmy S. Ren},
    title     = {Deep Bayesian Video Frame Interpolation},
    booktitle = {European Conference on Computer Vision (ECCV)},
    month     = {},
    year      = {2022},
    pages     = {}
}
```

### 4. Reference code base 
[[opencv_torchvision](https://github.com/hityzy1122/opencv_transforms_torchvision)], 
[[ESRGAN](https://github.com/xinntao/ESRGAN)], 
[[CAM-Net](https://github.com/niopeng/CAM-Net/tree/main/code)], 
[[SoftSplit](https://github.com/sniklaus/softmax-splatting)], 
[[DeepView](https://github.com/Findeton/deepview)], 
[[FLAVR](https://github.com/tarun005/FLAVR)], 
[[superSlomo](https://github.com/avinashpaliwal/Super-SloMo)], 
[[QVI](https://sites.google.com/view/xiangyuxu/qvi_nips19)]


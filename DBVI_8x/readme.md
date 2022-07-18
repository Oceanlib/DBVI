# 8x Interpolation
## 1. Requirements
1) cuda 9.0, cudnn7.6.5

2) python 3.6.9

3) pytorch 1.8.1

4) numpy 1.17.2

5) tqdm

6) gcc 5.4.0

7) cmake 3.16.0

8) opencv_contrib_python

9) Install apex: https://github.com/NVIDIA/apex

10) For distributed training with multi-gpus on cluster: slurm 15.08.11

## 2. Preparing Dataset
To get the training/testing datasets we used, you can download original frames from following links and preprocess using the code in ../mkDataset/forXXX/ by changing each 'Path/to/' accordingly, or directly download part of ready-to-use lmdb files we processed [here](https://pan.baidu.com/s/1meK6lCXrwrBQ3KFgos1aDw?pwd=2022)(password:2022)
#### Links:
[GoPro](https://drive.google.com/file/d/1rJTmM9_mLCNzBUUhYIGldBYgup279E_f/view), 
[X4K1000FPS](https://github.com/JihyongOh/XVFI#X4K1000FPS), 
[Adobe240(official)](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip), 
[Adobe240_lmdb(we used)](https://pan.baidu.com/s/1E5TAUAks_AzWEcmgwuR8oA?pwd=2022)(password:2022)

After processing the frames or downloading the provided lmdb files, put them in ./datasets. The path should be originized as:
```
datasets/
        GoPro/
             gopro_test_lmdb/
                             data.mdb
                             lock.mdb
                             sample.pkl
             gopro_train_lmdb/
                             data.mdb
                             lock.mdb
                             sample.pkl
 ```

## 3. Training


## 4. Testing with Pretrained Models

### For 8x interpolation
[Models](https://pan.baidu.com/s/1pxRFu29r56nDLgIHqFzHBA) pretrained on GoPro (password:2022)  
[Models](https://pan.baidu.com/s/1bXUaHN_n1F2YL8N9V5oMqw) pretrained on X4K1000FPS (password:2022)  
Unzip downloaded models and put them under ./output/


## 5. Citation 
```
@InProceedings{Yu_2022_ECCV,
    author    = {Yu, Zhiyang and Zhang, Yu and Xiang, Xujie and Zou, Dongqing and Chen, Xijun and Jimmy S. Ren},
    title     = {Deep Bayesian Video Frame Interpolation},
    booktitle = {European Conference on Computer Vision (ECCV)},
    month     = {},
    year      = {2022},
    pages     = {}
}
```

## 6. Reference code base 
[[ESRGAN](https://github.com/xinntao/ESRGAN)], 
[[SoftSplit](https://github.com/sniklaus/softmax-splatting)], 
[[DeepView](https://github.com/Findeton/deepview)], 
[[FLAVR](https://github.com/tarun005/FLAVR)], 
[[superSlomo](https://github.com/avinashpaliwal/Super-SloMo)], 
[[QVI](https://sites.google.com/view/xiangyuxu/qvi_nips19)]

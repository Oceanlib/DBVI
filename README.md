# Deep Bayesian Video Frame Interpolation
# README is under construction
[[Paper]](https://www.ecva.net/papers.php),  [[Video]](https://www.youtube.com/watch?v=8KvFwN1_3DY)

## Requirements

## Training


## Testing with Pretrained Models

### For 8x interpolation
[Models](https://pan.baidu.com/s/1pxRFu29r56nDLgIHqFzHBA) pretrained on GoPro (password:2022)  
[Models](https://pan.baidu.com/s/1bXUaHN_n1F2YL8N9V5oMqw) pretrained on X4K1000FPS (password:2022)  
Unzip downloaded models and put them under ./DBVI_8x/output/
### For 2x interpolation
[Model](https://pan.baidu.com/s/1TOtVA8f7my5vzB0n_kOEnA) pretrained on Vimeo-septulets (password:2022)  
Unzip the downloaded model and put it under ./DBVI_2x/output/


## Dataset
To get the training/testing datasets we used, you can download original frames from following links and preprocess using the code in ./mkDataset/forXXX/ by changing each 'Path/to/' accordingly, or directly download part of ready-to-use lmdb files we processed [here](https://pan.baidu.com/s/1meK6lCXrwrBQ3KFgos1aDw?pwd=2022)(password:2022)
### For 8x interpolation
[GoPro](https://drive.google.com/file/d/1rJTmM9_mLCNzBUUhYIGldBYgup279E_f/view), 
[X4K1000FPS](https://github.com/JihyongOh/XVFI#X4K1000FPS), 
[Adobe240(official)](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip), 
[Adobe240_lmdb(we used)](https://pan.baidu.com/s/1E5TAUAks_AzWEcmgwuR8oA?pwd=2022)(password:2022)
### For 2x interpolation
[Vimeo_Septuplet](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip), 
[ucf101](https://sites.google.com/view/xiangyuxu/qvi_nips19), 
[DAVIS](https://sites.google.com/view/xiangyuxu/qvi_nips19), 
[SNU-FILM](https://myungsub.github.io/CAIN/)

### 6. Citation 
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

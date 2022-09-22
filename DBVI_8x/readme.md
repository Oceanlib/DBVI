# 8x Interpolation
## 1. Preparing Dataset
The training/testing datasets we used can be either downloaded from following links and processed with the codes in ../mkDataset/forXXX/ by changing each 'Path/to/' accordingly, or directly downloaded from [here](https://pan.baidu.com/s/1meK6lCXrwrBQ3KFgos1aDw?pwd=2022)(password:2022)(the ready-to-use lmdb files)
#### Links:
[GoPro](https://drive.google.com/file/d/1rJTmM9_mLCNzBUUhYIGldBYgup279E_f/view), 
[X4K1000FPS](https://github.com/JihyongOh/XVFI#X4K1000FPS), 
[Adobe240(official)](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip), 
[Adobe240_lmdb(selected and used in paper)](https://pan.baidu.com/s/1E5TAUAks_AzWEcmgwuR8oA?pwd=2022)(password:2022)

The processed files should be put in ./datasets and originized as:
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

## 2. Training
### Training with single gpu:
(1) Set the name of train set(GoPro/X4K1000FPS), whether resume or not, dir of checkpoints and the name of pretrained weights(only needed if resume is true) in configs/configTrain.py(line50~54).

(2) Open a terminal and run ifconfig to get your ip address: XXX.XXX.XXX.XXX

(3) python train.py --initNode=XXX.XXX.XXX.XXX

### Distributed training with muli-gpus(16GPU,2Nodes) on cluser managed by [slurm](https://slurm.schedmd.com/quickstart_admin.html):
(1) Set the name of train set(GoPro/X4K1000FPS), whether resume or not, dir of checkpoints and the name of pretrained weights(only needed if resume is true) in configs/configTrain.py(line50~54).

(2) Set the name of part and nodes in cluser, number and index of gpus/cpus per-node and so on in runTrain.py(line3~14).

The example in runTrain.py is running on one part named Pixel, two nodes named 'SH-IDC1-10-5-39-55' and 'SH-IDC1-10-5-31-54', with 8 gpus per-node.

(3) python runTrain.py

## 4. Testing with Pretrained Models

[Models](https://pan.baidu.com/s/1pxRFu29r56nDLgIHqFzHBA) pretrained on GoPro (password:2022)  
[Models](https://pan.baidu.com/s/1bXUaHN_n1F2YL8N9V5oMqw) pretrained on X4K1000FPS (password:2022)  
Download models and put them under ./output/

### Testing with single gpu:
(1) Set the name of test set, dir of checkpoints and the name of pretrained weights in configs/configTest.py(line50~54).

(2) Open a terminal and run ifconfig to get your ip address: XXX.XXX.XXX.XXX

(3) python test.py --initNode=XXX.XXX.XXX.XXX

(4) PNG results on X4K1000FPS is provided [here](https://pan.baidu.com/s/1Quw5ToZ2itVmE-B0v6PKBA)(password:2022) for users with limited GPU memory (22GB at least). The results evaluated on quantized PNGs may be a little different from those reported in paper (less than 0.1db)

### Distributed testing with muli-gpus(10) on cluser managed by [slurm](https://slurm.schedmd.com/quickstart_admin.html):
(1) Set the name of test set, dir of checkpoints and the name of pretrained weights in configs/configTest.py(line50~54).

(2) Set the name of part and nodes in cluser, number and index of gpus/cpus per-node in runTest.py(line3~14).

The example in runTest.py is running on one part named Pixel with two nodes named 'SH-IDC1-10-5-39-55' and 'SH-IDC1-10-5-31-38' and 5 gpus per-node.

(3) python runTest.py


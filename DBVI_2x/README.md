# 2x Interpolation

## 1. Preparing Dataset
The training/testing datasets we used can be either downloaded from following links and processed with the codes in ../mkDataset/forXXX/ by changing each 'Path/to/' accordingly, or directly downloaded from [here](https://pan.baidu.com/s/1meK6lCXrwrBQ3KFgos1aDw?pwd=2022)(password:2022)(the ready-to-use lmdb files)
#### Links:
[Vimeo_Septuplet](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip), 
[ucf101](https://sites.google.com/view/xiangyuxu/qvi_nips19), 
[DAVIS](https://sites.google.com/view/xiangyuxu/qvi_nips19), 
[SNU-FILM](https://myungsub.github.io/CAIN/)

The processed files should be put in ./datasets and originized as:
```
datasets/
        vimeo/
             test_lmdb/
                      data.mdb
                      lock.mdb
                      sample.pkl
             train_lmdb/
                       data.mdb
                       lock.mdb
                       sample.pkl
 ```

## 2. Training
### Training with single gpu:
(1) Set whether resume or not, dir of checkpoints and the name of pretrained weights(only needed if resume is true) in configs/configTrain.py(line50~54).

(2) Open a terminal and run ifconfig to get your ip address: XXX.XXX.XXX.XXX

(3) python train.py --initNode=XXX.XXX.XXX.XXX

### Distributed training with muli-gpus(16GPU,2Nodes) on cluser managed by [slurm](https://slurm.schedmd.com/quickstart_admin.html):
(1) Set the name of train set(GoPro/X4K1000FPS), whether resume or not, dir of checkpoints and the name of pretrained weights(only needed if resume is true) in configs/configTrain.py(line50~54).

(2) Set the name of part and nodes in cluser, number and index of gpus/cpus per-node and so on in runTrain.py(line3~14).

The example in runTrain.py is running on one part named Pixel, two nodes named 'SH-IDC1-10-5-39-55' and 'SH-IDC1-10-5-31-54', with 8 gpus per-node.

(3) python runTrain.py

## 3. Testing with Pretrained Models
[Model](https://pan.baidu.com/s/1TOtVA8f7my5vzB0n_kOEnA) pretrained on Vimeo-septulets (password:2022)  
Download model and put it under ./output/

### Testing with single gpu:
(1) Set the name of test set, dir of checkpoints and the name of pretrained weights in configs/configTest.py(line50~55).

(2) Open a terminal and run ifconfig to get your ip address: XXX.XXX.XXX.XXX

(3) python test.py --initNode=XXX.XXX.XXX.XXX

### Distributed testing with muli-gpus(10) on cluser managed by [slurm](https://slurm.schedmd.com/quickstart_admin.html):
(1) Set the name of test set, dir of checkpoints and the name of pretrained weights in configs/configTest.py(line50~55).

(2) Set the name of part and nodes in cluser, number/index of gpus/cpus per-node and so on in runTest.py(line3~14).

The example in runTest.py is running on one part named Pixel with two nodes named 'SH-IDC1-10-5-39-55' and 'SH-IDC1-10-5-31-54' and 5 gpus per-node.

(3) python runTest.py


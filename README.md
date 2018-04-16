# Defense against Universal Adversarial Perturbations
This repository contains the Tensorflow implementation of ["Defense against Universal Adversarial Perturbations"(CVPR2018)](https://arxiv.org/abs/1711.05929)

<img src="https://github.com/liujianee/Pertrubation_Rectifying_Network/blob/master/assets/Teaser.png" width="50%">

## Environment
- Python 2.7
- Tensorflow 1.4.0

## Usage
### TESTING

1. Download [Inception PRN models](https://drive.google.com/drive/folders/1hP8l1vwCVCHfqKGOHu2Fyk_e9x5CpyoL?usp=sharing) from google drive.

2. Edit [TESTING script](https://github.com/liujianee/Pertrubation_Rectifying_Network/blob/master/Inception/TESTING_fooling_rate/TEST_SCRIPT.sh), set -`-data_dir` to the direcotry of clean images, set `--perturb_dir` to the base pertubation directory, and set `--pert_test_dir` to the TESTING perturbation folder.

Directory structure of `-data_dir`:

    ILSVRC2012_img
    ├── 1
    │   ├── ILSVRC2012_val_00000756.png
    │   ├── ILSVRC2012_val_00001260.png
    │   ├── ILSVRC2012_val_00006145.png
    │   └── ...
    ├── 2
    ├── 3
    ├── ...
    └── 1000
 
Directory structure of `--pert_test_dir`:

    inception_L2_Pert
    ├── perturbation_map_1.npy 
    ├── perturbation_map_2.npy 
    └── ...

4. Run [TESTING script](https://github.com/liujianee/Pertrubation_Rectifying_Network/blob/master/Inception/TESTING_fooling_rate/TEST_SCRIPT.sh).


### TRAINING

1. Edit [TRAINING configuration](https://github.com/liujianee/Pertrubation_Rectifying_Network/blob/master/Inception/config.py). Set `--data_dir` to clean images, note that `--data_dir_1`~`--data_dir_4` are the augmentation images. `perturb_dir` and `pert_train_dir` have similar reference as in TESTING.

2. Run [TRAINING script](https://github.com/liujianee/Pertrubation_Rectifying_Network/blob/master/Inception/TRAIN_SCRIPT.sh).


## References
- [LTS4 Universal project](https://github.com/LTS4/universal)
- [Taehoon Kim's simGAN code](https://github.com/carpedm20/simulated-unsupervised-tensorflow)



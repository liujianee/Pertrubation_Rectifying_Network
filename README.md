# Defense against Universal Adversarial Perturbations
This repository contains the Tensorflow implementation of ["Defense against Universal Adversarial Perturbations"(CVPR2018)](https://arxiv.org/abs/1711.05929)

<img src="https://github.com/liujianee/Pertrubation_Rectifying_Network/blob/master/assets/Teaser.png" width="50%">

# My environment
- Python 2.7
- Tensorflow 1.4.0

# Usage
### TESTING
- Download pre-trained [Inception PRN models](https://drive.google.com/drive/folders/1hP8l1vwCVCHfqKGOHu2Fyk_e9x5CpyoL?usp=sharing) from google drive.
- Edit TESTING [script](https://github.com/liujianee/Pertrubation_Rectifying_Network/blob/master/Inception/TESTING_fooling_rate/TEST_SCRIPT.sh), set --data_dir to the direcotry of clean images, and set --pert_test_dir to the directory of perturbation maps.
- Directory structure of clean images:




 
- Directory structure of perturbation maps:

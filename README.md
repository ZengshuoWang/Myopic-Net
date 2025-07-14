# Myopic-Net

By Zengshuo Wang, Haohan Zou, Yin Guo, Minghe Sun, Xin Zhao, Yan Wang, Mingzhu Sun.

This repository contains an official implementation of Myopic-Net for the paper "Myopic-Net: Deep Learning-Based Direct Identification of Myopia Onset and Progression", which has been accepted by Translational Vision Science & Technology (TVST).

## Quick start

### Environment

This code is developed on Python 3.8.5 and Pytorch 1.8.0 with NVIDIA GPUs. Training and testing are performed using 2 24G NVIDIA GeForce RTX 3090 GPUs with CUDA 11.1. Other platforms or GPUs are not fully tested.

### Install

1. Install Pytorch
2. Install dependencies

```shell
pip install -r requirements.txt
```

3. Because we choose FlowNetCorr as the backbone of Myopic-Net, which comes from the outstanding work ["FlowNet"](https://doi.org/10.1109/ICCV.2015.316). You should follow the instruction of the [pytorch implementation](https://github.com/NVIDIA/flownet2-pytorch) to ensure that FlowNetCorr can run properly. And there are mainly two steps:

   (1)
   ```shell
   cd Myopic-Net
   bash install.sh
   ```
   
   (2) Download the pre-trained model "FlowNet2-C[149MB]", and place it in the "models" folder.

4. 

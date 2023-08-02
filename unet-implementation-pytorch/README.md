# Unet Implementation PyTorch

Unet 主要用于图像分割问题。

本文将先简单介绍 Unet 的理论基础，然后使用 pytorch 一步一步地实现 Unet 图像分割。

## Data

- https://www.kaggle.com/c/tgs-salt-identification-challenge

## env

```
# conda create --name [ENV NAME] python=3.10

conda create --name unet-init python=3.10

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

pip install notebook numpy pandas opencv-python tqdm matplotlib chardet charset-normalizer  
```

## run the notebook:

```
# conda activate [ENV NAME]

conda activate unet-init

jupyter notebook
```


## Reference

- https://zhuanlan.zhihu.com/p/97488817

- https://link.zhihu.com/?target=https%3A//github.com/Qiuyan918/Unet_Implementation_PyTorch/blob/master/Unet_Implementation_PyTorch.ipynb



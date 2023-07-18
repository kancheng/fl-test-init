
# Medicine MNIST Dataset

醫療醫學影像處理


## Package

```
pip install -r requirements.txt

pip install notebook
```
- matplotlib==3.7.2
- numpy==1.25.0
- opencv_python==4.8.0.74
- scikit_learn==1.3.0
- seaborn==0.12.2
- torch==2.0.1
- torchvision==0.15.2
- tqdm==4.65.0

## nvcc

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_May__3_18:49:52_PDT_2022
Cuda compilation tools, release 11.7, V11.7.64
Build cuda_11.7.r11.7/compiler.31294372_0
```

## Directory Structure

數據下載後的結構

```
├── input
│   ├── medical_mnist
│   │   ├── AbdomenCT [10000 entries exceeds filelimit, not opening dir]
│   │   ├── BreastMRI [8954 entries exceeds filelimit, not opening dir]
│   │   ├── ChestCT [10000 entries exceeds filelimit, not opening dir]
│   │   ├── CXR [10000 entries exceeds filelimit, not opening dir]
│   │   ├── Hand [10000 entries exceeds filelimit, not opening dir]
│   │   └── HeadCT [10000 entries exceeds filelimit, not opening dir]
│   └── medical_mnist_processed
│       ├── test
│       │   ├── AbdomenCT [1000 entries exceeds filelimit, not opening dir]
│       │   ├── BreastMRI [895 entries exceeds filelimit, not opening dir]
│       │   ├── ChestCT [1000 entries exceeds filelimit, not opening dir]
│       │   ├── CXR [1000 entries exceeds filelimit, not opening dir]
│       │   ├── Hand [1000 entries exceeds filelimit, not opening dir]
│       │   └── HeadCT [1000 entries exceeds filelimit, not opening dir]
│       ├── train
│       │   ├── AbdomenCT [7000 entries exceeds filelimit, not opening dir]
│       │   ├── BreastMRI [6267 entries exceeds filelimit, not opening dir]
│       │   ├── ChestCT [7000 entries exceeds filelimit, not opening dir]
│       │   ├── CXR [7000 entries exceeds filelimit, not opening dir]
│       │   ├── Hand [7000 entries exceeds filelimit, not opening dir]
│       │   └── HeadCT [7000 entries exceeds filelimit, not opening dir]
│       └── valid
│           ├── AbdomenCT [2000 entries exceeds filelimit, not opening dir]
│           ├── BreastMRI [1790 entries exceeds filelimit, not opening dir]
│           ├── ChestCT [2000 entries exceeds filelimit, not opening dir]
│           ├── CXR [2000 entries exceeds filelimit, not opening dir]
│           ├── Hand [2000 entries exceeds filelimit, not opening dir]
│           └── HeadCT [2000 entries exceeds filelimit, not opening dir]
├── notebooks
│   └── process_medical_mnist.ipynb
├── outputs
│   ├── accuracy.png
│   ├── heatmap.png
│   ├── loss.png
│   ├── model.pth
│   ├── test_image_1998.png
│   ├── test_image_2997.png
│   ├── test_image_3996.png
│   ├── test_image_4995.png
│   └── test_image_999.png
└── src
    ├── datasets.py
    ├── model.py
    ├── test.py
    ├── train.py
    └── utils.py
```

## Train & Test

```
python train.py --epochs 20
python test.py
```

## Reference

- https://github.com/apolanco3225/Medical-MNIST-Classification

- https://www.kaggle.com/datasets/andrewmvd/medical-mnist

- https://debuggercafe.com/training-a-custom-pytorch-classifier-on-medical-mnist-dataset/



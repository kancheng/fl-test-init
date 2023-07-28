
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

```
(sim-med) kan@kan-MS-7E07:~/new-test/tuberculosis-medicine/classifier-on-tuberculosis-dataset$ tree -f -L 3
.
├── ./input
│   ├── ./input/xray-dataset
│   │   ├── ./input/xray-dataset/cim-normal
│   │   └── ./input/xray-dataset/cim-tuberculosis
│   └── ./input/xray-dataset-processed
│       ├── ./input/xray-dataset-processed/test
│       ├── ./input/xray-dataset-processed/train
│       └── ./input/xray-dataset-processed/valid
├── ./notebooks
│   └── ./notebooks/process_medical_tuberculosis.ipynb
├── ./outputs
│   ├── ./outputs/accuracy.png
│   ├── ./outputs/heatmap.png
│   ├── ./outputs/loss.png
│   └── ./outputs/model.pth
└── ./src
    ├── ./src/datasets.py
    ├── ./src/model.py
    ├── ./src/__pycache__
    │   ├── ./src/__pycache__/datasets.cpython-310.pyc
    │   ├── ./src/__pycache__/model.cpython-310.pyc
    │   └── ./src/__pycache__/utils.cpython-310.pyc
    ├── ./src/test.py
    ├── ./src/train.py
    └── ./src/utils.py

12 directories, 13 files
(sim-med) kan@kan-MS-7E07:~/new-test/tuberculosis-medicine/classifier-on-tuberculosis-dataset$ 

```

## Reference

- https://github.com/apolanco3225/Medical-MNIST-Classification

- https://www.kaggle.com/datasets/andrewmvd/medical-mnist

- https://debuggercafe.com/training-a-custom-pytorch-classifier-on-medical-mnist-dataset/



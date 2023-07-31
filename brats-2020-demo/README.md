# BraTS2020 Demo

測試 BraTS2020 Demo

## Dependencies

```
* Python 3.10
```

## Data Prepare

- https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

- https://www.kaggle.com/datasets/awsaf49/brats2020-training-data

```
├── code
│   └── brats-2020-pytorch-execution.ipynb
├── input
│   └── brats20-dataset-training-validation
│       ├── BraTS2020_TrainingData
│       │   └── MICCAI_BraTS2020_TrainingData
│       │       ├── BraTS20_Training_001
│       │       ├─   ....
│       │       ├── BraTS20_Training_369
│       │       ├── name_mapping.csv
│       │       └── survival_info.csv
│       └── BraTS2020_ValidationData
│           └── MICCAI_BraTS2020_ValidationData
│               ├── BraTS20_Validation_001
│       │       ├─   ....
│               ├── BraTS20_Validation_125
│               ├── name_mapping_validation_data.csv
│               └── survival_evaluation.csv

```

## 目錄結構

訓練完後的目錄結構。

```
.
├── code
│   ├── best_model.pth
│   ├── brats-2020-pytorch-execution.ipynb
│   ├── last_epoch_model.pth
│   ├── result1.png
│   ├── result1.svg
│   ├── train_data.csv
│   └── train_log.csv
├── img
│   ├── result1.png
│   └── result1.svg
├── input
│   └── brats20-dataset-training-validation
│       ├── BraTS2020_TrainingData
│       │   └── MICCAI_BraTS2020_TrainingData
│       │       ├── BraTS20_Training_001
│       │       ├─   ....
│       │       ├── BraTS20_Training_369
│       │       ├── name_mapping.csv
│       │       └── survival_info.csv
│       └── BraTS2020_ValidationData
│           └── MICCAI_BraTS2020_ValidationData
│               ├── BraTS20_Validation_001
│       │       ├─   ....
│               ├── BraTS20_Validation_125
│               ├── name_mapping_validation_data.csv
│               └── survival_evaluation.csv
└── README.md
```

## Reference

- https://medium.com/@sumitgulati59/brain-tumor-segmentation-b97de6619e04

- https://github.com/LauraMoraB/BrainTumorSegmentation

- https://github.com/mtancak/PyTorch-UNet-Brain-Cancer-Segmentation

- https://github.com/akhanss/BraTS-2020

- https://towardsdatascience.com/review-v-net-volumetric-convolution-biomedical-image-segmentation-aa15dbaea974

- https://github.com/Sumit-github08/Brain-Tumor-Segmentation

- https://www.kaggle.com/code/mojtabanafez/brats-2020-pytorch-execution/notebook

- https://www.kaggle.com/code/arashmehrzadi/brain-tumor-segmentation-unet

- https://github.com/raja21068/Federated-Learning-For-Medical-Images

- https://github.com/lescientifik/open_brats2020



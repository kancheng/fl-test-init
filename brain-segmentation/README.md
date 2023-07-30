# brain-segmentation

## Details

針對 Brain_Segmentation 所進行的實驗範例使用 QuickNAT，數據使用 OASIS-1。原範例為 inesdv26/Brain-Segmentation 修改。

## Tools

```
pip install notebook

jupyter notebook
```

## Directory Structure

執行後目錄狀況。

```
(nii) kan@kan-MS-7E07:~/new-test/brain-segmentation$ tree -f -L 1
.
├── ./best_model_coronal
├── ./Brain_Segmentation_using_QuickNAT_bu.ipynb
├── ./Brain_Segmentation_using_QuickNAT.ipynb
├── ./checkpoints_coronal
├── ./oasis_cross-sectional_disc1.tar.gz
├── ./Oasis_Data
├── ./Oasis_Data_Processed
├── ./Oasis_Labels_Processed
├── ./README.md
├── ./TestData
├── ./TestLabels
├── ./TrainData
├── ./TrainLabels
├── ./ValData
└── ./ValLabels

11 directories, 4 files
(nii) kan@kan-MS-7E07:~/new-test/brain-segmentation$ 
```

## Reference

- Data were provided by OASIS-1: Cross-Sectional: Principal Investigators: D. Marcus, R, Buckner, J, Csernansky J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382.

- https://github.com/inesdv26/Brain-Segmentation

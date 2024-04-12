# AMLSII_23-24_SN23201000
ELEC 0135 Applied Machine Learning Systems II aimed at solving the NTIRE 2017 super resolution challenge

## 1. Prerequisites
To Begin with, it is required to download the datasets and put it into the empty Datasets folder. The structure of the Datasets folder are as follows:

* Datasets/
  * DIV2K_train_HR/
  * DIV2K_train_LR_bicubic_X2/  DIV2K_train_LR_bicubic/X2
  * DIV2K_train_LR_unknown/DIV2K_train_LR_unknown/X2
  * DIV2K_valid_HR/
  * DIV2K_valid_LR_bicubic_X2/DIV2K_valid_LR_bicubic/X2
  * DIV2K_valid_LR_unknown/DIV2K_valid_LR_unknown/X2

The folder Datasets is not included, you can either download the required Datesets according to the file tree or you can download the zip file [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/). 

### The environment

My advice is to create a new conda environment by the following command first:

```bash
conda create -n myenv python=3.10.14
```
Then, enter the AMLSII_23-24_SN23201000 folder, activate the environment created above and install all the package that required for this project by:

```bash
pip install -r requirements.txt
```

## 2. How to check the result of this project

Just run the main.py

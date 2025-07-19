# SGVI-2023

This project integrates multi-source remote sensing imagery and street view images to demonstrate how the Green View Index (GVI) can be effectively estimated using remote sensing data alone.


# 🔧 Usage


## 1. Download Data and Pre-trained Model

Download required datasets and pre-trained model weights. Make sure to store them in folders with the same names as referenced in the code.

## 2. Train the Model

The training pipeline is implemented in train.py. It demonstrates how to train a deep learning model (ReViT) using multi-source remote sensing imagery and GVI ground truth labels extracted from street view images.

## 3. Apply the Model

The inference procedure is provided in apply.py. This script shows how to use the trained ReViT model with remote sensing data to estimate GVI at a large spatial scale and across different seasons.




# 📌 Notes


Datasets and model weights are hosted on Figshare.
(Download link: coming soon)

The file normalization_stats.txt contains the maximum and minimum values for each band of the multi-source remote sensing imagery. These values were computed from the entire raw dataset and are used to perform min-max normalization during training and inference.

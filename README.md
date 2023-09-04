# Deep Learning-based Inversion Model

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Results](#results)
- [License](./LICENSE)

# Overview

Lunar surface chemistry is essential for revealing petrological characteristics needed to understand the geological evolution of the Moon. A series of lunar surface chemical compositions have been mapped with related abundances from Apollo and Luna returned samples. However, this mapping could only calibrate chemical features before 3.0 Gyr, missing the critical second half of the history of the Moon. Therefore, young lunar soil samples with different chemical characteristics are necessary for mapping surface chemistry comprehensively. Here we present new major oxides chemistry maps of major oxides by adding 2.0 Gyr Chang’e-5 samples consisting in a new type of differentiated lunar basaltic rock carrying distinctive chemical contents compared with Apollo and Luna samples. The introduction of Chang'e 5 samples in combination with a deep learning-based inversion model resulted in inferred chemical contents in the new maps that are more precise than the Lunar Prospector Gamma-Ray Spectrometer (GRS) maps and are closest to the abundances of current returned samples when compared to the existing literature. Meanwhile, the verification of in situ measurement data acquired by Chang'e 3 and Chang'e 4 lunar rover demonstrated that Chang’e-5 samples are indispensable ground truth in mapping lunar surface chemistry. From the new maps, the molar or atom ratio of Mg/(Mg+Fe) is recalculated thus refining the division of lunar geologic units. A young mare basalts unit is determined based on new inferred compositions. This identifies critical potential sites to constrain the late lunar magmatic and thermal history.

# Repo Contents

- [datasets](./datasets) : the data adopted to train the inversion model.
- [model_ckpt](./model_ckpt) : the weights of the inversion model used for oxides abundance estimation on the lunar surface.
- [spec_Kaguya_MI_by_row](./spec_Kaguya_MI_by_row) : the example data for prediction.
- [utils](./utils) : tools to be used in the project.
- [models.py](./models.py) : inversion model.
- [trainer.py](./trainer.py) : model training process.
- [configs.yaml](./configs.yaml) : base configuration file for model training.
- [main_training.py](./main_training.py) : main code for training inversion model.
- [main_predicting.py](./main_predicting.py) : main code for predicting oxides abundance on the lunar surface.


# System Requirements

## Hardware Requirements

The inversion model requires only a standard computer. For the model training phase, a standard computer is sufficient. Of course a computer with a Graphic Processing Unit (GPU)  can increase the training speed. For the model prediction phase, huge amounts of data need to be processed to predict the oxide abundance on the lunar surface, therefore, computers with GPU are recommended for improving efficiency. In our study, we used a PC workstation with the following specs:
  `RAM` : 128 GB  
  `CPU` : Intel(R) Xeon(R) Platinum 8352Y CPU @ 2.20GHz
  `GPU` : NVIDIA GeForce RTX 3090 Graphics Processing Unit

## Software Requirements

- The code has been tested on the following  OS systems:
  - `Linux` : Ubuntu 20.04
  - `Windows` : Windows 10

- The code is implemented in `Python` and `Python >= 3.6` is recommended.


# Installation Guide

Users only need to clone this code and install the relevant dependency packages, that is, complete the code installation. The detail operations are as follows:

```
git clone https://github.com/hszhaohs/DL-IM
cd DL-IM
pip install -r requirements.txt
```


# Demo

## Traning with the leave-one-out cross-validation (LOOCV) configuration

The TiO2 inversion model is trained using the following commands:
```
python main_training.py --gpu 0 --dataname TiO2
```
The training commands for the other oxide inversion models are similar.

## Predicting the oxide abundance on the lunar surface

The TiO2 abundance on the lunar surface is predicted using the following commands:
```
python main_predicting.py --gpu 0 --dataname TiO2 --datapath ./spec_Kaguya_MI_by_row
```
The predicting commands for the other oxides are similar.


# Results

Under the LOOCV setting, we achieved better performance than the comparison method on all six oxides (Supplementary Table 3). The introduction of Chang'e 5 samples in combination with a deep learning-based inversion model resulted in inferred chemical contents in the new maps that are more precise than the Lunar Prospector Gamma-Ray Spectrometer (GRS) maps and are closest to the abundances of current returned samples when compared to the existing literature.


# License

`DL-IM` is free software made available under the MIT License. For details see the `LICENSE.md` file.

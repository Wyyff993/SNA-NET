#SNA-Nnet

## Introduction
This paper presents a new solution for low-light image enhancement by collectively exploiting Signal-to-Noise-Ratio-aware transformers and convolutional models to dynamically enhance pixels with spatial-varying operations.
They are long-range operations for image regions of extremely low Signal-to-Noise-Ratio (SNR) and short-range operations for other regions. 
We propose to take an SNR prior to guide the feature fusion and formulate the SNR-aware transformer with a new selfattention model to avoid tokens from noisy image regions of very low SNR.

## dataset

### LOL datasets
The directory of LOL-v2-real should contain Real_captured/Train and Real_captured/Test.

The directory of LOL-v2-synthetic should contain Synthetic/Train and Synthetic/Test.

Please place the directory of LOL-v1, LOL-v2-real, and LOL-v2-synthetic under "datasets"

### SDSD dataset
Different from original SDSD datasets with dynamic scenes, we utilize its static version (the scenes are the same of original SDSD).

And you can download the SDSD-indoor and SDSD-outdoor from [baidu pan](https://pan.baidu.com/s/1rfRzshGNcL0MX5soRNuwTA) (验证码: jo1v) and [baidu pan](https://pan.baidu.com/s/1JzDQnFov-u6aBPPgjSzSxQ) (验证码: uibk), and there should contain "indoor_static_np" and "outdoor_static_np".
They can also be downloaded from [google pan](https://drive.google.com/drive/folders/14TF0f9YQwZEntry06M93AMd70WH00Mg6?usp=share_link)

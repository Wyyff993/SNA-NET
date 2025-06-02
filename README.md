## Signal-to-Noise Ratio Guided Noise Adaptive Network via Dual-Domain Collaboration for Low-Light Image Enhancement

## dataset

### LOL datasets
The directory of LOL-v2-real should contain Real_captured/Train and Real_captured/Test.

The directory of LOL-v2-synthetic should contain Synthetic/Train and Synthetic/Test.

### SDSD dataset
The evaluation setting could follow the following descriptions: randomly select 12 scenes from indoor subset and take others as the training data. The performance on indoor scene is computed on the first 30 frames in each of this 12 scenes, i.e., 360 frames.

The arrangement of the dataset is  
--indoor/outdoor  
----GT (the videos under normal light)  
--------pair1  
--------pair2  
--------...  
----LQ (the videos under low light)  
--------pair1  
--------pair2  
--------...  

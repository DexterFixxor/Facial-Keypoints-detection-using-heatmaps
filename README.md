# Facial keypoints detection using Convolutional Pose Estimation


This repository contains code for training neural network using Pytorch framework. Neural network model is based on [Convolutional Pose Machines](https://arxiv.org/abs/1602.00134) paper, but instead of body parts network is trained to recognize facial keypoints and output probability map for each keypoint.

Dataset used is [Facial Keypints (68)](https://www.kaggle.com/tarunkr/facial-keypoints-68-dataset). It consists of two folders, _test_ and _train_, and .csv files with information about each image and it's keypoints locations. During training, _train_ folder is split in two parts (split size defined in _config.py_). One for training and the other for validation. Test folder can be used after training is complete to test model once more.


# Training

1. Download dataset and place it inside project folder.
2. Inside _config.py_ file change **DATA_ROOT_PATH** to coresponding folder name. For example: "data". Rest of the file leave at default, change _batch size_ and _number of workers_ depending on hardware used.
3. Run _main.py_ to start training. Model, loss graph and copy of configuration parameters is saved in "output/current_date_and_time" folder in project root. 
   Along with it, examples of some iterations in each epoch is saved, also each new _best_ model.

Example of folder structure:

```
├── ...
├── data/ 
│   ├── test/
│   ├── training/
|   ├── ...
└── ...
```

Trained model can be downloaded from [this link](https://drive.google.com/drive/folders/1rChgc3t1EJa3vlEkCc0BEswvgPLEanXV?usp=sharing). Model was trained on image resolution of 320x320.

## Test model using webcam
Inside _camera_test.py_ change:

```
...
# path to the trained model
model_load_path = './weights/cpm_net_ep60.pt' 
...
```

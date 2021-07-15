import math

import numpy as np
import pandas as pd
import os
from config import cfg
import matplotlib.pyplot as plt
import cv2


def create_folder(path=None):
    if path is not None:
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

def save_config_params():
    if not os.path.isdir(os.path.dirname(cfg.CONFIG_SAVE)):
        os.makedirs(os.path.dirname(cfg.CONFIG_SAVE))
    if not os.path.isdir(os.path.dirname(f"{cfg.OUTPUT_PATH}/heatmap_results/")):
        os.makedirs(os.path.dirname(f"{cfg.OUTPUT_PATH}/heatmap_results/"))
    with open(cfg.CONFIG_SAVE, 'w') as f:
        f.write('\n' + '=' * 50 + '\n')
        #print([name for name, value in vars(cfg).items()])
        f.write('\n'.join('%s: %s' %item for item in vars(cfg).items()))
        f.write('\n' + '=' * 50 + '\n')


def split_data(csv_path, portion=0.1):
    df_data = pd.read_csv(csv_path)
    len_data = len(df_data)
    # calculate the validation data sample length
    valid_split = int(len_data * portion)
    # calculate the training data samples length
    train_split = int(len_data - valid_split)
    training_samples = df_data.iloc[:train_split][:]
    valid_samples = df_data.iloc[-valid_split:][:]

    return training_samples, valid_samples


def get_max_width_and_height(csv_data, image_folder):
    max_w, max_h = 0, 0
    for i in range(len(csv_data)):
        image = cv2.imread(f"{image_folder}/{csv_data.iloc[i][0]}")
        max_w = max(max_w, image.shape[0])
        max_h = max(max_h, image.shape[1])

    return max_w, max_h


def show_keypoints(img, keypoints):
    plt.imshow(img)
    plt.scatter(keypoints[:,0], keypoints[:,1], s=20, marker='.', c='m')


def gaussian_filter(x, y, H, W, sigma=5):
    channel = [math.exp(-((c - x) ** 2 + (r - y) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
    channel = np.array(channel, dtype=np.float32)
    channel = np.reshape(channel, newshape=(H, W))

    return channel


def gaussian_filter_2(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x)**2 + (gridy - center_y) ** 2
    return np.exp(-D2/2.0/sigma/sigma)


if __name__ == "__main__":
    plt.plot()

import time

import torch
from datetime import datetime
import os
import time
from collections import OrderedDict


class Config():

    def __init__(self):
        self.ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
        self.DATA_ROOT_PATH = os.path.join(self.ROOT_DIR, 'data')
        self.OUTPUT_PATH = os.path.join(self.ROOT_DIR, 'output/{}'.format(datetime.now().strftime("%d-%m-%Y-%H-%M")) )
        self.CONFIG_SAVE = os.path.join(self.OUTPUT_PATH, 'config.txt')

        # Learning params
        self.NUM_WORKERS = 4
        self.BATCH_SIZE = 8
        self.LR = 0.0001
        self.EPOCH = 60
        self.VALID_EACH = 1
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.TEST_SPLIT = 0.1
        self.IMG_SIZE = 150     # for rescaling all images to this size
        self.IMG_CROP = 120     # this is value fed into the network (make it at least 10px smaller then IMG_SIZE, max 30)
        self.MAX_IMG_W = 150    # used to calculate padding
        self.MAX_IMG_H = 150    # if no padding is needed, set to same as IMG_CROP
        self.HEATMAP_STRIDE = 2  # possible choices: 2, 4
        self.HEATMAP_WEIGHT = (self.IMG_SIZE * self.IMG_SIZE / 1.0) / (self.HEATMAP_STRIDE ** 2)

        self.USE_RootMSE = True # if False, MSE is used


cfg = Config()

if __name__ == "__main__":
    for name, value in cfg.CONV_DICT.items():
        print(value)
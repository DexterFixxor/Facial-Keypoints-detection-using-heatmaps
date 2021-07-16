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
        self.BATCH_SIZE = 32
        self.LR = 0.00002
        self.EPOCH = 40
        self.VALID_EACH = 1
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.TEST_SPLIT = 0.1
        self.IMG_SIZE = 200     # must be divisonable with 8
        self.MAX_IMG_W = 256
        self.MAX_IMG_H = 256
        self.HEATMAP_STRIDE = 4  # possible choices: 2, 4
        self.HEATMAP_WEIGHT = cfg.IMG_SIZE * cfg.IMG_SIZE * 68 / 1.0

        self.USE_RootMSE = True # if False, MSE is used


cfg = Config()

if __name__ == "__main__":
    for name, value in cfg.CONV_DICT.items():
        print(value)
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
        self.NUM_WORKERS = 8
        self.BATCH_SIZE = 16  # possible 32 if stride=8, bcs of more memory in GPU
        self.LR = 1e-5  # default 0.0001
        self.EPOCH = 60
        self.VALID_EACH = 1
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.TEST_SPLIT = 0.1
        # Rescale images to this size (tuple: fixed img size, int: keep ratio), should be less than MAX_IMG_SIZE
        self.IMG_SIZE = 320
        # Crop resolution (type must be tuple)
        self.IMG_CROP = (320, 320)
        # If type(IMG_SIZE) == int, images are zero padded around border to this size
        self.MAX_IMG_SIZE = (400, 400)
        # heatmap stride possible choices: 2, 4
        self.HEATMAP_STRIDE = 8
        self.SIGMA = 1
        __img_size_1, __img_size_2 = (self.IMG_SIZE, self.IMG_SIZE) if isinstance(self.IMG_SIZE, int) else self.IMG_SIZE
        self.HEATMAP_WEIGHT = 1.0  # (__img_size_1 * __img_size_2 * 68/ 1.0) / (self.HEATMAP_STRIDE ** 2)

        self.optimizer = 'AdamW'


cfg = Config()

if __name__ == "__main__":
    for name, value in cfg.CONV_DICT.items():
        print(value)
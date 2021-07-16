import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import cfg
from src import utils, MyTransforms
from src.cpm_model import CPM
from src.dataset import FaceKeypointsDataset

import time


def delay_sec(sec):
    start = time.time()
    while (time.time() - start) < sec:
        pass

def get_keypoints(net, in_image):
    _, _, _, _, _, heatmap6 = net(in_image)
    heatmap6 = heatmap6.detach().cpu().numpy()
    kp_list = []

    fig, axes = plt.subplots(2, 8)
    for x in range(2):
        for y in range(8):
            hm = heatmap6[0, x+y+1, :, :]
            axes[x, y].imshow(hm)

    fig.tight_layout()
    for i in range(68):
        hm = cv2.resize(hm, (cfg.IMG_SIZE, cfg.IMG_SIZE))

        _, conf, _, point = cv2.minMaxLoc(hm)
        x = point[0]
        y = point[1]
        kp_list.append((int(x), int(y), conf))
    plt.show()
    return kp_list


def draw_keypoints(img, kpts):

    for i in range(len(kpts)):
        if kpts[i][2] > 0.5 and i != 0:
            cv2.circle(img, kpts[i][:2], [255, 0, 0], -1, cv2.LINE_AA)

    return img


if __name__ == "__main__":

    train_samples, valid_samples = utils.split_data(csv_path=f"{cfg.DATA_ROOT_PATH}/training_frames_keypoints.csv",
                                                    portion=cfg.TEST_SPLIT)

    img_folder = f"{cfg.DATA_ROOT_PATH}/training"
    model_load_path = './output/15-07-2021-23-43/cpm_net.pt'

    model = CPM(n_keypoints=68)
    model.load_state_dict(torch.load(model_load_path))

    model.to(cfg.DEVICE)
    model.eval()

    resize = MyTransforms.Resize(max_img_size=256, desired_img_size=128)

    for i in range(len(valid_samples)):
        img = cv2.imread(f"{img_folder}/{train_samples.iloc[i][0]}")
        img, _ = resize(img, None)
        h, w, c = img.shape
        top, left = (256 - h)//2, (256-w)//2
        img_padded = np.zeros((256, 256, 3), dtype='float32')
        img_padded[top:top + h, left:left + w] = img
        img_padded = img_padded / 255.

        plt.imshow(img_padded)
        plt.show()

        img_padded = np.transpose(img_padded, (2,0,1))
        img_padded = torch.FloatTensor(img_padded)
        img_padded = torch.unsqueeze(img_padded, 0)
        img_padded = img_padded.cuda()

        key_points = get_keypoints(model, img_padded)
        #frame = draw_keypoints(img, key_points)

        #cv2.imshow('frame',frame)
        #cv2.waitKey(10000)


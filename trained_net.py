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

    for i in range(68):
        hm = heatmap6[0, i+1, :, :]
        #hm = cv2.resize(hm, (cfg.IMG_SIZE, cfg.IMG_SIZE))

        _, conf, _, point = cv2.minMaxLoc(hm)
        x = point[0] * cfg.HEATMAP_STRIDE
        y = point[1] * cfg.HEATMAP_STRIDE
        kp_list.append((int(x), int(y), conf))

    return kp_list


def draw_keypoints(img, kpts):

    for i in range(len(kpts)):
        if  i != 0:
            cv2.circle(img, kpts[i][:2], 1, [255, 0, 0], -1, cv2.LINE_AA)

    return img


if __name__ == "__main__":

    train_samples, valid_samples = utils.split_data(csv_path=f"{cfg.DATA_ROOT_PATH}/training_frames_keypoints.csv",
                                                    portion=cfg.TEST_SPLIT)

    img_folder = f"{cfg.DATA_ROOT_PATH}/training"
    model_load_path = './output/19-07-2021-00-48/models/cpm_net_ep60.pt'

    model = CPM(n_keypoints=68)
    model.load_state_dict(torch.load(model_load_path))

    model.to(cfg.DEVICE)
    model.eval()

    resize = MyTransforms.Resize(desired_img_size=cfg.IMG_SIZE)

    for i in range(len(valid_samples)):
        img = cv2.imread(f"{img_folder}/{valid_samples.iloc[i][0]}")
        img, _ = resize(img, None)
        h, w, c = img.shape
        top, left = (cfg.MAX_IMG_SIZE[0] - h)//cfg.HEATMAP_STRIDE, (cfg.MAX_IMG_SIZE[1] - w)//cfg.HEATMAP_STRIDE
        img_padded = np.zeros((cfg.MAX_IMG_SIZE[0], cfg.MAX_IMG_SIZE[1], c), dtype='float32')
        img_padded[top:top + h, left:left + w] = img
        img_padded = img_padded / 255.

        #plt.imshow(img_padded)
        #plt.show()

        img_padded = np.transpose(img_padded, (2,0,1))
        img_padded_tensor = torch.FloatTensor(img_padded)
        img_padded_unsqueeze = torch.unsqueeze(img_padded_tensor, 0)
        img_padded_cuda = img_padded_unsqueeze.cuda()

        key_points = get_keypoints(model, img_padded_cuda)
        frame = draw_keypoints(img_padded.transpose(1,2,0), key_points)

        plt.imshow(frame)
        plt.show()
        #cv2.waitKey(10000)


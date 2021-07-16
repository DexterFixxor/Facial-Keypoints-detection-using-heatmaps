import torch
import cv2
import numpy as np
from config import cfg
from src import utils

from src.cpm_model import CPM
from src.dataset import FaceKeypointsDataset


def get_keypoints(net, input):

    _, _, _, _, _, heatmap6 = net(input)

    kp_list = []

    for i in range(69):

        hm = heatmap6[0, i, :, :]
        hm = cv2.resize(hm, (cfg.IMG_SIZE, cfg.IMG_SIZE))

        _, conf, _, point = cv2.minMaxLoc(hm)
        x = point[0]
        y = point[1]
        kp_list.append((int(x), int(y), conf))


def draw_keypoints(img, kpts):

    for i in len(kpts):
        if kpts[i][2] > 0.5:
            cv2.circle(img, kpts[i][:2], [255, 0, 0], -1, cv2.LINE_AA)

    return img


if __name__ == "__main__":

    train_samples, valid_samples = utils.split_data(csv_path=f"{cfg.DATA_ROOT_PATH}/training_frames_keypoints.csv",
                                                    portion=cfg.TEST_SPLIT)

    img_folder = f"{cfg.DATA_ROOT_PATH}/training"
    model_load_path = '/output/15-07-2021-23-43/cpm_net.pt'

    model = CPM(n_keypoints=68)
    model.load_state_dict(torch.load(model_load_path))

    model.to(cfg.DEVICE)
    model.eval()

    for i in range(len(valid_samples)):
        img = cv2.imread(f"{img_folder}/{valid_samples.iloc[i][0]}")
        img = img / 255.  # Normalize image
        img = torch.FloatTensor(img)
        img = img.to(cfg.DEVICE)

        key_points = get_keypoints(model, img)
        frame = draw_keypoints(img, key_points)

        cv2.imshow(frame)
        cv2.waitKey(10000)


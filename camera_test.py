import time

import numpy as np
import cv2
import torch

from config import cfg
from src import utils, MyTransforms
from src.cpm_model import CPM

import matplotlib.pyplot as plt


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
            cv2.circle(img, kpts[i][:2], 1, [255, 255, 0], -1, cv2.LINE_AA)
    return img


if __name__ == "__main__":

    train_samples, valid_samples = utils.split_data(csv_path=f"{cfg.DATA_ROOT_PATH}/training_frames_keypoints.csv",
                                                    portion=cfg.TEST_SPLIT)

    img_folder = f"{cfg.DATA_ROOT_PATH}/training"
    model_load_path = './weights/cpm_net_ep60.pt'

    model = CPM(n_keypoints=68)
    model.load_state_dict(torch.load(model_load_path))

    model.to(cfg.DEVICE)
    model.eval()

    resize = MyTransforms.Resize(desired_img_size=cfg.IMG_SIZE)

    # ---- CAMERA ----
    cap = cv2.VideoCapture(0)
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 5.0, cfg.IMG_CROP)
    if not cap.isOpened():
        print("\nCannot open camera\n")
        exit()

    while True:
        start = time.time()
        ret, frame = cap.read()

        if not ret:
            print("Can't receive stream. Exiting...")
            break
        else:
            img, _ = resize(frame, None)
            h, w, c = img.shape

            top, left = (cfg.MAX_IMG_SIZE[0] - h), (cfg.MAX_IMG_SIZE[1] - w)
            top, left = top // 2, left // 2
            img_padded = np.zeros((cfg.MAX_IMG_SIZE[0], cfg.MAX_IMG_SIZE[1], c), dtype='float32')
            img_padded[top:top + h, left:left + w] = img
            img_padded = img_padded / 255.

            x_center, y_center = cfg.MAX_IMG_SIZE[1] // 2,  cfg.MAX_IMG_SIZE[0] // 2
            x_start, x_stop = x_center - cfg.IMG_CROP[1]//2, x_center + cfg.IMG_CROP[1]//2
            y_start, y_stop = y_center - cfg.IMG_CROP[0]//2, y_center + cfg.IMG_CROP[0]//2

            img_padded = img_padded[y_start : y_stop, x_start : x_stop]

            img_padded = np.transpose(img_padded, (2, 0, 1))
            img_padded_tensor = torch.FloatTensor(img_padded)
            img_padded_unsqueeze = torch.unsqueeze(img_padded_tensor, 0)
            img_padded_cuda = img_padded_unsqueeze.cuda()

            key_points = get_keypoints(model, img_padded_cuda)
            frame_kp = draw_keypoints(img_padded.transpose(1, 2, 0), key_points)

            cv2.imshow('frame', frame_kp)
            out.write(np.uint8(frame_kp*255))
            # print(f"FPS: {1/(time.time() - start)}")
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

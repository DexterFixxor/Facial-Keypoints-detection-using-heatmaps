import time

import torch
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import cv2

from config import cfg
import src.utils as utils
import src.MyTransforms as mytf


class FaceKeypointsDataset(Dataset):

    def __init__(self, csv_data, img_folder, img_size, padding_size, crop_size:tuple):
        self.data = csv_data
        self.img_folder = img_folder
        self.img_size = img_size
        self.padding_size = padding_size
        self.crop_size = crop_size

        self.stride = cfg.HEATMAP_STRIDE  # 2 = upola smanjuje rezoluciju slike

        # 1 premalo, povecava vreme izvrsavanja znatno,
        # 5 je previse, ne dolaze do izrazaja detalji sa lica, poklapaju se
        self.sigma = 1

        self.resize = mytf.Resize(desired_img_size=self.img_size)

        self.transform = torchvision.transforms.Compose([
            mytf.ImgPadding(padding_size=self.padding_size, stride=self.stride),
            # Pads to the border
            mytf.RandomCrop(output_size=self.crop_size, stride=self.stride),
            mytf.ToTensor(),
        ])

    def __getitem__(self, index):
        start = time.time()
        img = cv2.imread(f"{self.img_folder}/{self.data.iloc[index][0]}")
        img = img / 255.  # Normalize image

        kpts = self.data.iloc[index][1:]
        kpts = np.array(kpts, dtype='float32').reshape(-1, 2)

        #TODO: add random flip?
        image, keypoints = self.resize(img, kpts)
        #h, w, c = image.shape

        # image.shape returns Rows x Columns x Channels                                               +1 for background
        heatmaps = np.zeros((image.shape[0] // self.stride, image.shape[1] // self.stride, keypoints.shape[0] + 1), dtype='float32')

        for k, (x, y) in enumerate(keypoints):
            # Don't show values not in image but labeled in keypoints
            x = int(x * 1.0 / self.stride)
            y = int(y * 1.0 / self.stride)
            heatmap = utils.gaussian_filter_2(size_h=image.shape[0]//self.stride, size_w=image.shape[1]//self.stride,
                                              center_x=x, center_y=y, sigma=self.sigma)
            heatmap[heatmap > 1] = 1
            heatmap[heatmap < 0.0099] = 0
            heatmaps[:, :, k+1] = heatmap

        output = {'image': image, 'heatmaps': heatmaps}
        output = self.transform(output)
        output['heatmaps'][0, :, :] = torch.from_numpy(1.0 - np.max(np.asarray(output['heatmaps'][1:, :, :]), axis=0))

        #print(f"{index}: Image shape: {output['image'].shape} ::: Heatmaps shape: {output['heatmaps'].shape}")
        #print("----------------")
        #print("Time: {}".format(time.time() - start))
        #plt.imshow(output['image'].detach().cpu().numpy().transpose(1,2,0))
        #plt.imshow(cv2.resize(np.asarray(output['heatmaps'][0,:,:]), (cfg.IMG_CROP, cfg.IMG_CROP)), alpha=0.5)
        #plt.show()
        return output

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    train_samples, valid_samples = utils.split_data(csv_path=f"{cfg.DATA_ROOT_PATH}/training_frames_keypoints.csv",
                                                    portion=cfg.TEST_SPLIT)

    train_dataset = FaceKeypointsDataset(csv_data=train_samples,
                                         img_folder=f"{cfg.DATA_ROOT_PATH}/training",
                                         img_size=cfg.IMG_SIZE,
                                         padding_size=cfg.MAX_IMG_SIZE,
                                         crop_size=cfg.IMG_CROP)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
                              num_workers=cfg.NUM_WORKERS,
                              shuffle=True, drop_last=True)

    iterTrain = iter(train_dataset)
    start = time.time()
    for i, sample in enumerate(train_loader):

        #sample = next(iterTrain)
        time.sleep(1)
        #print(time.time() - start)
        start = time.time()
        pass
        # plt.figure('not padded')
        # plt.imshow(sample['image'])
        # plt.show()
        # padTransform = mytf.ImgPadding(max_w=cfg.MAX_IMG_W, max_h=cfg.MAX_IMG_H)
        # cropTrans = mytf.RandomCrop(output_size=(cfg.IMG_SIZE, cfg.IMG_SIZE))

        # padSample = padTransform(sample)
        # print(f"Shape of padded heatmaps is: {padSample['heatmaps'].shape}")

        # cropSample = cropTrans(padSample)
        # print(f"Shape of heatmaps is: {sample['heatmaps'].shape}")
        # plt.figure('crop')
        # plt.imshow(sample['image'])
        # plt.imshow(sample['heatmaps'][:,:, 0], alpha=0.5)
        # plt.show()

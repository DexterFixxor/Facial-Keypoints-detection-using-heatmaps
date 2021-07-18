import cv2
import torch
import numpy as np
import time

from matplotlib import pyplot as plt


class ToTensor(object):

    def __call__(self, sample):
        image, heatmaps = sample['image'], sample['heatmaps']

        if len(image.shape) == 3:
            image = image.transpose(2, 0, 1)

        image = torch.from_numpy(image)
        if len(image.shape) == 2:
            image = torch.unsqueeze(image, 0)

        heatmaps = heatmaps.transpose(2, 0, 1)
        heatmaps = torch.from_numpy(heatmaps)

        return {'image': image.float(), 'heatmaps': heatmaps.float()}


class ImgPadding(object):

    def __init__(self, padding_size, stride):
        self.max_h, self.max_w = padding_size[0], padding_size[1]
        self.stride = stride

    def __call__(self, sample):
        img, hm = sample['image'], sample['heatmaps']

        h, w, img_depth = img.shape
        hm_h, hm_w, hm_depth = hm.shape

        max_h = max(self.max_h, h)
        max_w = max(self.max_w, w)
        top_offset = abs(max_h - h) // 2
        left_offset = abs(max_w - w) // 2
        image = np.zeros((max_h, max_w, img_depth), dtype='float32')
        image[top_offset:top_offset + h, left_offset:left_offset + w] = img

        top_offset = abs(max_h // self.stride - hm_h) // 2
        left_offset = abs(max_w // self.stride - hm_w) // 2
        heatmaps = np.zeros((max_h // self.stride, max_w//self.stride, hm_depth), dtype='float32')
        heatmaps[top_offset:top_offset + hm_h, left_offset:left_offset + hm_w] = hm

        return {'image': image, 'heatmaps': heatmaps}


class RandomCrop(object):

    def __init__(self, output_size, stride):
        self.output_size = output_size
        self.stride = stride

    def __call__(self, sample):
        image, hm = sample['image'], sample['heatmaps']

        h, w = image.shape[0], image.shape[1]

        if isinstance(self.output_size, int):
            new_h, new_w = self.output_size, self.output_size
        else:
            new_h = self.output_size[0]
            new_w = self.output_size[1]

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        # Rescaling because of downsampling
        heatmaps = hm[
                   top // self.stride:(top + new_h)//self.stride,
                   left // self.stride:(left + new_w)//self.stride,
                   :]

        return {'image': image, 'heatmaps': heatmaps}


class Resize(object):

    def __init__(self, desired_img_size):
        self.desired_size = desired_img_size

    def __call__(self, image, keypoints):
        h, w = image.shape[:2]

        #if h >= self.max_size or w >= self.max_size:
        if isinstance(self.desired_size, int):
            if h > w:
                new_h, new_w = self.desired_size * h / w, self.desired_size
            else:
                new_h, new_w = self.desired_size, self.desired_size * w / h
        else:
            new_h, new_w = self.desired_size

        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_w, new_h))

        # scale the pts, too
        if keypoints is not None:
            key_pts = keypoints * [new_w / w, new_h / h]

            return img, key_pts

        return img, None


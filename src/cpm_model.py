import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg

def conv_output(img_size, kernel, padding=1, stride=1):
    return (img_size - kernel + 2*padding)/stride +1


def maxpool_output(input_size, padding, kernel_size, stride):
    return (input_size + 2*padding - kernel_size)/stride +1


class CPM(nn.Module):

    def __init__(self, n_keypoints):
        super(CPM, self).__init__()
        self.kp = n_keypoints
        """ STAGE 1 """
        self.stage1_conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(9, 9), padding=(4, 4))
        self.stage1_pool1 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)  # Reduces img resolution x2
        self.stage1_conv2 = nn.Conv2d(128, 128, (9,9), (1,1), (4,4))
        if cfg.HEATMAP_STRIDE == 2:
            self.stage1_pool2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        else:
            self.stage1_pool2 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        self.stage1_conv3 = nn.Conv2d(128, 128, (9,9), (1,1), (4,4))
        self.stage1_pool3 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.stage1_conv4 = nn.Conv2d(128, 32, (5, 5), (1, 1), (2, 2))

        self.stage1_conv5 = nn.Conv2d(32, 512, (9, 9), (1, 1), (4, 4))
        self.stage1_conv6 = nn.Conv2d(512, 512, kernel_size=(1, 1))
        self.stage1_conv7 = nn.Conv2d(512, self.kp + 1, kernel_size=(1, 1))

        """ STAGE 2 a) --- feature extraction"""
        self.stage2_conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(9, 9), padding=(4, 4))
        self.stage2_pool1 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)  # Reduces img resolution x2
        self.stage2_conv2 = nn.Conv2d(128, 128, (9, 9), (1, 1), (4, 4))
        if cfg.HEATMAP_STRIDE == 2:
            self.stage2_pool2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        else:
            self.stage1_pool2 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        self.stage2_conv3 = nn.Conv2d(128, 128, (9, 9), (1, 1), (4, 4))
        self.stage2_pool3 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.stage2_conv4 = nn.Conv2d(128, 32, (5, 5), (1, 1), (2, 2))

        """ STAGE 2 b) --- classification"""
        self.Mconv1_stage2 = nn.Conv2d(32 + self.kp + 1, 128, kernel_size=(11, 11), padding=(5, 5))
        self.Mconv2_stage2 = nn.Conv2d(128, 128, kernel_size=(11, 11), padding=(5, 5))
        self.Mconv3_stage2 = nn.Conv2d(128, 128, kernel_size=(11, 11), padding=(5, 5))
        self.Mconv4_stage2 = nn.Conv2d(128, 128, kernel_size=(1, 1), padding=(0, 0))
        self.Mconv5_stage2 = nn.Conv2d(128, self.kp+ 1, kernel_size=(1, 1), padding=(0, 0))

        """ STAGE 3 b)"""
        self.Mconv1_stage3 = nn.Conv2d(32 + self.kp + 1, 128, kernel_size=(11, 11), padding=(5, 5))
        self.Mconv2_stage3 = nn.Conv2d(128, 128, kernel_size=(11, 11), padding=(5, 5))
        self.Mconv3_stage3 = nn.Conv2d(128, 128, kernel_size=(11, 11), padding=(5, 5))
        self.Mconv4_stage3 = nn.Conv2d(128, 128, kernel_size=(1, 1), padding=(0, 0))
        self.Mconv5_stage3 = nn.Conv2d(128, self.kp + 1, kernel_size=(1, 1), padding=(0, 0))

        """ STAGE 4 b)"""
        self.Mconv1_stage4 = nn.Conv2d(32 + self.kp + 1, 128, kernel_size=(11, 11), padding=(5, 5))
        self.Mconv2_stage4 = nn.Conv2d(128, 128, kernel_size=(11, 11), padding=(5, 5))
        self.Mconv3_stage4 = nn.Conv2d(128, 128, kernel_size=(11, 11), padding=(5, 5))
        self.Mconv4_stage4 = nn.Conv2d(128, 128, kernel_size=(1, 1), padding=(0, 0))
        self.Mconv5_stage4 = nn.Conv2d(128, self.kp + 1, kernel_size=(1, 1), padding=(0, 0))

        """ STAGE 5 b)"""
        self.Mconv1_stage5 = nn.Conv2d(32 + self.kp + 1, 128, kernel_size=(11, 11), padding=(5, 5))
        self.Mconv2_stage5 = nn.Conv2d(128, 128, kernel_size=(11, 11), padding=(5, 5))
        self.Mconv3_stage5 = nn.Conv2d(128, 128, kernel_size=(11, 11), padding=(5, 5))
        self.Mconv4_stage5 = nn.Conv2d(128, 128, kernel_size=(1, 1), padding=(0, 0))
        self.Mconv5_stage5 = nn.Conv2d(128, self.kp + 1, kernel_size=(1, 1), padding=(0, 0))

        """ STAGE 6 b)"""
        self.Mconv1_stage6 = nn.Conv2d(32 + self.kp + 1, 128, kernel_size=(11, 11), padding=(5, 5))
        self.Mconv2_stage6 = nn.Conv2d(128, 128, kernel_size=(11, 11), padding=(5, 5))
        self.Mconv3_stage6 = nn.Conv2d(128, 128, kernel_size=(11, 11), padding=(5, 5))
        self.Mconv4_stage6 = nn.Conv2d(128, 128, kernel_size=(1, 1), padding=(0, 0))
        self.Mconv5_stage6 = nn.Conv2d(128, self.kp + 1, kernel_size=(1, 1), padding=(0, 0))

    def _stage1(self, image):
        x = self.stage1_pool1(F.relu(self.stage1_conv1(image)))
        x = self.stage1_pool2(F.relu(self.stage1_conv2(x)))
        x = self.stage1_pool3(F.relu(self.stage1_conv3(x)))
        x = F.relu(self.stage1_conv4(x))
        x = F.relu(self.stage1_conv5(x))
        x = F.relu(self.stage1_conv6(x))
        x = F.relu(self.stage1_conv7(x))

        return x

    def _stage2_feature_extract(self, image):
        x = self.stage2_pool1(F.relu(self.stage2_conv1(image)))
        x = self.stage2_pool2(F.relu(self.stage2_conv2(x)))
        x = self.stage2_pool3(F.relu(self.stage2_conv3(x)))
        x = F.relu(self.stage2_conv4(x))
        return x

    def _stage2_cnn(self, stage1, features):
        x = torch.cat([stage1, features], 1)
        x = F.relu(self.Mconv1_stage2(x))
        x = F.relu(self.Mconv2_stage2(x))
        x = F.relu(self.Mconv3_stage2(x))
        x = F.relu(self.Mconv4_stage2(x))
        x = self.Mconv5_stage2(x)
        return x

    def _stage3_cnn(self, stage2, features):
        x = torch.cat([stage2, features], 1)
        x = F.relu(self.Mconv1_stage3(x))
        x = F.relu(self.Mconv2_stage3(x))
        x = F.relu(self.Mconv3_stage3(x))
        x = F.relu(self.Mconv4_stage3(x))
        x = self.Mconv5_stage3(x)
        return x

    def _stage4_cnn(self, stage3, features):
        x = torch.cat([stage3, features], 1)
        x = F.relu(self.Mconv1_stage4(x))
        x = F.relu(self.Mconv2_stage4(x))
        x = F.relu(self.Mconv3_stage4(x))
        x = F.relu(self.Mconv4_stage4(x))
        x = self.Mconv5_stage4(x)
        return x

    def _stage5_cnn(self, stage4, features):
        x = torch.cat([stage4, features], 1)
        x = F.relu(self.Mconv1_stage5(x))
        x = F.relu(self.Mconv2_stage5(x))
        x = F.relu(self.Mconv3_stage5(x))
        x = F.relu(self.Mconv4_stage5(x))
        x = self.Mconv5_stage5(x)
        return x

    def _stage6_cnn(self, stage5, features):
        x = torch.cat([stage5, features], 1)
        x = F.relu(self.Mconv1_stage6(x))
        x = F.relu(self.Mconv2_stage6(x))
        x = F.relu(self.Mconv3_stage6(x))
        x = F.relu(self.Mconv4_stage6(x))
        x = self.Mconv5_stage6(x)
        return x

    def forward(self, image):
        stage1_heatmap = self._stage1(image)

        extracted_features_stage2 = self._stage2_feature_extract(image)

        stage2_heatmap = self._stage2_cnn(stage1_heatmap, extracted_features_stage2)
        stage3_heatmap = self._stage3_cnn(stage2_heatmap, extracted_features_stage2)
        stage4_heatmap = self._stage4_cnn(stage3_heatmap, extracted_features_stage2)
        stage5_heatmap = self._stage5_cnn(stage4_heatmap, extracted_features_stage2)
        stage6_heatmap = self._stage6_cnn(stage5_heatmap, extracted_features_stage2)

        return stage1_heatmap, stage2_heatmap, stage3_heatmap, stage4_heatmap, stage5_heatmap, stage6_heatmap


if __name__ == "__main__":

    conv1 = conv_output(480, 9, 4)
    max1 = maxpool_output(conv1, kernel_size=2, padding=0, stride=2)
    conv2 = conv_output(max1, 9, 4)
    max2 = maxpool_output(conv2, padding=1, kernel_size=3, stride=1)
    conv3 = conv_output(max2, 9, 4)
    max3 = maxpool_output(conv3, padding=1, kernel_size=3, stride=1)

    conv4 = conv_output(max3, kernel=5, padding=2, stride=1)
    conv5 = conv_output(conv4, kernel=9, padding=4, stride=1)
    conv6 = conv_output(conv5, kernel=1, padding=0, stride=1)
    conv7 = conv_output(conv6, kernel=1, padding=0, stride=1)



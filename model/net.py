import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size1=5, kernel_size2=5):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size2)
        self.batchnorm1 = nn.BatchNorm2d(mid_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu((self.batchnorm1(self.conv1(x))))
        x = F.relu((self.batchnorm2(self.conv2(x))))
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size1=5, kernel_size2=5):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.transconv1 = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=kernel_size1)
        self.transconv2 = nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=kernel_size2)
        self.batchnorm1 = nn.BatchNorm2d(mid_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu((self.batchnorm1(self.transconv1(x))))
        x = F.relu((self.batchnorm2(self.transconv2(x))))
        return x


class MyNet(nn.Module):
    def __init__(self, n_channel):
        super(MyNet, self).__init__()
        self.n_channel = n_channel

        self.downconv1 = DownConv(n_channel, 15, 5, 50, 40)
        # (1, 15, 424, 424)
        self.maxpool1 = nn.MaxPool2d((2, 2))
        # (1, 15, 212, 212)
        self.downconv2 = DownConv(15, 500, 100, 30, 20)
        # (1, 500, 164, 164)
        self.maxpool2 = nn.MaxPool2d((2, 2))
        # (500, 82, 82)
        self.conv = nn.Conv2d(500, 500, 33)
        # (500, 50, 50)

        self.upsample1 = nn.Upsample(scale_factor=2)
        # (500, 100, 100)
        self.upconv1 = UpConv(500, 15, 100, 41, 61)
        # (15, 200, 200)
        self.upsample2 = nn.Upsample(scale_factor=2)
        # (15, 400, 400)
        self.upconv2 = UpConv(15, n_channel, 5, 62, 52)

    def forward(self, x):
        # 前向卷积
        x = self.conv(self.maxpool2(self.downconv2(self.maxpool1(self.downconv1(x)))))
        # 反向卷积
        x = self.upconv2(self.upsample2(self.upconv1(self.upsample1(x))))
        return x

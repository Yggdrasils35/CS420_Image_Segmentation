import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size1=5, kerneal_size2=5):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=kerneal_size2)
        self.batchnorm1 = nn.BatchNorm2d(mid_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu((self.batchnorm1(self.conv1(x))))
        x = F.relu((self.batchnorm2(self.conv2(x))))
        return x


class MyNet(nn.Module):
    def __init__(self, n_channel):
        super(MyNet, self).__init__()
        self.n_channel = n_channel

        self.downconv1 = DownConv(n_channel, 15, 5, 50, 40)
        self.maxpool1 = nn.MaxPool2d(2)
        self.downconv2 = DownConv(15, 500, 100, 30, 20)
        self.maxpool2 = nn.MaxPool2d(2)

        # Todo finished the reconvolution part

    def forward(self, x):
        return F.relu(self.downconv1(x))



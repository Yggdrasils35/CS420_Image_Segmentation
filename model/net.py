import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class DenseLayer(nn.Module):
    def __init__(self,in_channels,bottleneck_size,growth_rate):
        super(DenseLayer,self).__init__()
        count_of_1x1 = bottleneck_size
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(in_channels,count_of_1x1,kernel_size=1)

        self.bn2 = nn.BatchNorm2d(count_of_1x1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3x3 = nn.Conv2d(count_of_1x1,growth_rate,kernel_size=3,padding=1)
        
    def forward(self,*prev_features):
        # for f in prev_features:
        #     print(f.shape)

        input = torch.cat(prev_features,dim=1)
        # print(input.device,input.shape)
        # for param in self.bn1.parameters():
        #     print(param.device)
        # print(list())
        bottleneck_output = self.conv1x1(self.relu1(self.bn1(input)))
        out = self.conv3x3(self.relu2(self.bn2(bottleneck_output)))
        
        return out
class DenseBlock(nn.Module):
    def __init__(self,in_channels,layer_counts,growth_rate):
        super(DenseBlock,self).__init__()
        self.layer_counts = layer_counts
        self.layers = []
        for i in range(layer_counts):
            curr_input_channel = in_channels + i*growth_rate
            bottleneck_size = 4*growth_rate #论文里设置的1x1卷积核是3x3卷积核的４倍.
            layer = DenseLayer(curr_input_channel,bottleneck_size,growth_rate).cuda()       
            self.layers.append(layer)

    def forward(self,init_features):
        features = [init_features]
        for layer in self.layers:
            layer_out = layer(*features) #注意参数是*features不是features
            features.append(layer_out)

        return torch.cat(features, 1)
class DownTransitionBlock(nn.Module):
    def __init__(self,out_channel):
        super().__init__()
        self.maxP=nn.MaxPool2d(2,stride=2)
        self.bn=nn.BatchNorm2d(out_channel)   
        self.drop=nn.Dropout(p=0.2)
    def forward(self,x):
        out=self.maxP(x)
        out=self.bn(out)
        out=F.relu(out)
        out=self.drop(out)
        return out

class UpTransitionBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.upsample=nn.Upsample(scale_factor=2,mode='nearest')
        self.bn1=nn.BatchNorm2d(in_channel)
        self.conv=nn.Conv2d(in_channel,out_channel,1,1,0)
        self.bn2=nn.BatchNorm2d(out_channel)
        self.drop=nn.Dropout(p=0.2)
    def forward(self,x):
        out=self.upsample(x)
        out=self.bn1(out)
        out=F.relu(out)
        out=self.conv(out)
        out=self.bn2(out)
        out=F.relu(out)
        out=self.drop(out)
        return out

class Bottleneck(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.con1=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,padding=0,bias=False)
        self.drop=nn.Dropout(p=0.2)
        self.bn=nn.BatchNorm2d(out_channel)
    def forward(self,x):
        out=self.con1(x)
        out=self.bn(out)
        out=F.relu(out)
        out=self.drop(out)
        return out



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


class DenseUnet(nn.Module):
    def __init__(self):
        super(DenseUnet,self).__init__()
        self.Conv1=nn.Conv2d(1,8,3,1,1)
        self.DB1=DenseBlock(8,4,2)
        self.DTB1=DownTransitionBlock(16)
        self.DB2=DenseBlock(16,4,4)
        self.DTB2=DownTransitionBlock(32)
        self.DB3=DenseBlock(32,4,8)
        self.DTB3=DownTransitionBlock(64)
        self.DB4=DenseBlock(64,4,16)
        self.UTB1=UpTransitionBlock(128,64)
        self.BN1=Bottleneck(128,64)
        self.DB5=DenseBlock(64,4,16)
        self.BN2=Bottleneck(128,64)
        self.UTB2=UpTransitionBlock(64,32)
        self.BN3=Bottleneck(64,32)
        self.DB6=DenseBlock(32,4,8)
        self.BN4=Bottleneck(64,32)
        self.UTB3=UpTransitionBlock(32,16)
        self.BN5=Bottleneck(32,16)
        self.DB7=DenseBlock(16,4,4)
        self.BN6=Bottleneck(32,16)
        self.Conv2=nn.Conv2d(16,2,3,1,1)
        self.Conv3=nn.Conv2d(2,1,1,1)

    def forward(self,x):
        Res1=self.DB1(self.Conv1(x))
        Res2=self.DB2(self.DTB1(Res1))
        Res3=self.DB3(self.DTB2(Res2))
        out=self.UTB1(self.DB4(self.DTB3(Res3)))
        out=torch.cat([out,Res3],dim=1)
        Res3=None
        out=self.UTB2(self.BN2(self.DB5(self.BN1(out))))
        out=torch.cat([out,Res2],dim=1)
        Res2=None
        out=self.UTB3(self.BN4(self.DB6(self.BN3(out))))
        out=torch.cat([out,Res1],dim=1)
        Res1=None
        out=self.Conv3(self.Conv2(self.BN6(self.DB7(self.BN5(out)))))
        out=torch.sigmoid(out)
        return out

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
        x=torch.sigmoid(x)
        return x

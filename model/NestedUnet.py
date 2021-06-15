import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

class conv_block_nested(nn.Module):
    def __init__(self,in_channels,mid_channels,out_channels):
        super().__init__()
        
        self.conv1=nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1,bias=True)
        self.bn1=nn.BatchNorm2d(mid_channels)
        self.conv2=nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1,bias=True)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        return out

class NestedUNet(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        
        num=64
        filter_list=[num,num*2,num*4,num*8,num*16]

        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.Up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)

        self.conv0_0=conv_block_nested(in_channels,filter_list[0],filter_list[0])
        self.conv1_0=conv_block_nested(filter_list[0],filter_list[1],filter_list[1])
        self.conv2_0=conv_block_nested(filter_list[1],filter_list[2],filter_list[2])
        self.conv3_0=conv_block_nested(filter_list[2],filter_list[3],filter_list[3])
        self.conv4_0=conv_block_nested(filter_list[3],filter_list[4],filter_list[4])

        self.conv0_1 = conv_block_nested(filter_list[0] + filter_list[1], filter_list[0], filter_list[0])
        self.conv1_1 = conv_block_nested(filter_list[1] + filter_list[2], filter_list[1], filter_list[1])
        self.conv2_1 = conv_block_nested(filter_list[2] + filter_list[3], filter_list[2], filter_list[2])
        self.conv3_1 = conv_block_nested(filter_list[3] + filter_list[4], filter_list[3], filter_list[3])

        self.conv0_2 = conv_block_nested(filter_list[0]*2 + filter_list[1], filter_list[0], filter_list[0])
        self.conv1_2 = conv_block_nested(filter_list[1]*2 + filter_list[2], filter_list[1], filter_list[1])
        self.conv2_2 = conv_block_nested(filter_list[2]*2 + filter_list[3], filter_list[2], filter_list[2])

        self.conv0_3 = conv_block_nested(filter_list[0]*3 + filter_list[1], filter_list[0], filter_list[0])
        self.conv1_3 = conv_block_nested(filter_list[1]*3 + filter_list[2], filter_list[1], filter_list[1])

        self.conv0_4 = conv_block_nested(filter_list[0]*4 + filter_list[1], filter_list[0], filter_list[0])

        self.final = nn.Conv2d(filter_list[0], out_channels, kernel_size=1)
    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output
        
# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/8 15:31
# @Author : liumin
# @File : LEDnet.py

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

def ConvBNReLU(in_channels,out_channels,kernel_size,stride,padding,dilation=[1,1],groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation,groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


def ConvBN(in_channels,out_channels,kernel_size,stride,padding,dilation=[1,1],groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation,groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )

def ConvReLU(in_channels,out_channels,kernel_size,stride,padding,dilation=[1,1],groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation,groups=groups, bias=False),
            nn.ReLU6(inplace=True)
        )

def Conv1x1BNReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

class HalfSplit(nn.Module):
    def __init__(self, dim=1):
        super(HalfSplit, self).__init__()
        self.dim = dim

    def forward(self, input):
        splits = torch.chunk(input, 2, dim=self.dim)
        return splits[0], splits[1]

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


class SS_nbt(nn.Module):
    def __init__(self, channels, dilation=1, groups=4):
        super(SS_nbt, self).__init__()

        mid_channels = channels // 2
        self.half_split = HalfSplit(dim=1)

        self.first_bottleneck = nn.Sequential(
            ConvReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[3, 1], stride=1, padding=[1, 0]),
            ConvBNReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[1, 3], stride=1, padding=[0, 1]),
            ConvReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[3, 1], stride=1, dilation=[dilation,1], padding=[dilation, 0]),
            ConvBNReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[1, 3], stride=1, dilation=[1,dilation], padding=[0, dilation]),
        )

        self.second_bottleneck = nn.Sequential(
            ConvReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[1, 3], stride=1, padding=[0, 1]),
            ConvBNReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[3, 1], stride=1, padding=[1, 0]),
            ConvReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[1, 3], stride=1, dilation=[1,dilation], padding=[0, dilation]),
            ConvBNReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[3, 1], stride=1, dilation=[dilation,1], padding=[dilation, 0]),
        )

        self.channelShuffle = ChannelShuffle(groups)

    def forward(self, x):
        x1, x2 = self.half_split(x)
        x1 = self.first_bottleneck(x1)
        x2 = self.second_bottleneck(x2)
        out = torch.cat([x1, x2], dim=1)
        return self.channelShuffle(out+x)


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampling, self).__init__()
        mid_channels = out_channels - in_channels

        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=mid_channels,kernel_size=3,stride=2,padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2, padding=1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.maxpool(x)
        output = torch.cat([x1, x2], 1)
        return self.relu(self.bn(output))

class Encoder(nn.Module):
    def __init__(self, groups = 4):
        super(Encoder, self).__init__()
        planes = [32, 64, 128]

        self.downSampling1 = DownSampling(in_channels=3, out_channels=planes[0])
        self.ssBlock1 = self._make_layer(channels=planes[0], dilation=1, groups=groups, block_num=3)
        self.downSampling2 = DownSampling(in_channels=32, out_channels=planes[1])
        self.ssBlock2 = self._make_layer(channels=planes[1], dilation=1, groups=groups, block_num=2)
        self.downSampling3 = DownSampling(in_channels=planes[1], out_channels=planes[2])
        self.ssBlock3 = nn.Sequential(
            SS_nbt(channels=planes[2], dilation=1, groups=groups),
            SS_nbt(channels=planes[2], dilation=2, groups=groups),
            SS_nbt(channels=planes[2], dilation=5, groups=groups),
            SS_nbt(channels=planes[2], dilation=9, groups=groups),
            SS_nbt(channels=planes[2], dilation=2, groups=groups),
            SS_nbt(channels=planes[2], dilation=5, groups=groups),
            SS_nbt(channels=planes[2], dilation=9, groups=groups),
            SS_nbt(channels=planes[2], dilation=17, groups=groups),
        )

    def _make_layer(self, channels, dilation, groups, block_num):
        layers = []
        for idx in range(block_num):
            layers.append(SS_nbt(channels, dilation=dilation, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.downSampling1(x)
        x = self.ssBlock1(x)
        x = self.downSampling2(x)
        x = self.ssBlock2(x)
        x = self.downSampling3(x)
        out = self.ssBlock3(x)
        return out


class APN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(APN, self).__init__()

        self.conv1_1 = ConvBNReLU(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = Conv1x1BNReLU(in_channels=in_channels, out_channels=out_channels)

        self.conv2_1 = ConvBNReLU(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=2, padding=2)
        self.conv2_2 = Conv1x1BNReLU(in_channels=in_channels, out_channels=out_channels)

        self.conv3 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=in_channels, kernel_size=7, stride=2, padding=3),
            Conv1x1BNReLU(in_channels=in_channels, out_channels=out_channels),
        )

        self.conv1 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1),
            Conv1x1BNReLU(in_channels=in_channels,out_channels=out_channels),
        )

        self.branch2 = Conv1x1BNReLU(in_channels=in_channels, out_channels=out_channels)

        self.branch3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=1, stride=1,padding=0),
        )

    def forward(self, x):
        _, _, h, w = x.shape
        x1 = self.conv1_1(x)
        x2 = self.conv2_1(x1)
        x3 = self.conv3(x2)
        x3 = F.interpolate(x3, size=(h//4, w//4), mode='bilinear', align_corners=True)
        x2 = self.conv2_2(x2) + x3
        x2 = F.interpolate(x2, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        x1 = self.conv1_2(x1) + x2
        out1 = F.interpolate(x1, size=(h, w), mode='bilinear', align_corners=True)
        
        out2 = self.branch2(x)

        out3 = self.branch3(x)
        out3 = F.interpolate(out3, size=(h, w), mode='bilinear', align_corners=True)
        return out1 * out2 + out3


class Decoder(nn.Module):
    def __init__(self, in_channels,num_classes):
        super(Decoder, self).__init__()
        self.apn = APN(in_channels=in_channels, out_channels=num_classes)

    def forward(self, x):
        _, _, h, w = x.shape
        apn_x = self.apn(x)
        out = F.interpolate(apn_x, size=(h*8, w*8), mode='bilinear', align_corners=True)
        return out


class LEDnet(nn.Module):
    def __init__(self, num_classes=20):
        super(LEDnet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(in_channels=128,num_classes=num_classes)

    def forward(self, x):
        x = self.encoder(x)
        out = self.decoder(x)
        return out


if __name__ == '__main__':
    model = LEDnet(num_classes=20)
    print(model)

    input = torch.randn(1,3,1024,512)
    output = model(input)
    print(output.shape)
# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/10/13 8:52
# @Author : liumin
# @File : segnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


__all__ = ["SegNet"]


def Conv3x3BNReLU(in_channels,out_channels,stride,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, reverse=False):
        super().__init__()
        if reverse:
            self.double_conv = nn.Sequential(
                Conv3x3BNReLU(in_channels, in_channels, stride=1),
                Conv3x3BNReLU(in_channels, out_channels, stride=1)
            )
        else:
            self.double_conv = nn.Sequential(
                Conv3x3BNReLU(in_channels, out_channels,stride=1),
                Conv3x3BNReLU(out_channels, out_channels, stride=1)
            )

    def forward(self, x):
        return self.double_conv(x)


class TripleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 3"""
    def __init__(self, in_channels, out_channels, reverse=False):
        super().__init__()
        if reverse:
            self.triple_conv = nn.Sequential(
                Conv3x3BNReLU(in_channels, in_channels, stride=1),
                Conv3x3BNReLU(in_channels, in_channels, stride=1),
                Conv3x3BNReLU(in_channels, out_channels, stride=1)
            )
        else:
            self.triple_conv = nn.Sequential(
                Conv3x3BNReLU(in_channels, out_channels,stride=1),
                Conv3x3BNReLU(out_channels, out_channels, stride=1),
                Conv3x3BNReLU(out_channels, out_channels, stride=1)
            )

    def forward(self, x):
        return self.triple_conv(x)


class SegNet(nn.Module):
    """
        SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
        https://arxiv.org/pdf/1511.00561.pdf
    """
    def __init__(self,classes= 19):
        super(SegNet, self).__init__()

        self.conv_down1 = DoubleConv(3, 64)
        self.conv_down2 = DoubleConv(64, 128)
        self.conv_down3 = TripleConv(128, 256)
        self.conv_down4 = TripleConv(256, 512)
        self.conv_down5 = TripleConv(512, 512)

        self.conv_up5 = TripleConv(512, 512, reverse=True)
        self.conv_up4 = TripleConv(512, 256, reverse=True)
        self.conv_up3 = TripleConv(256, 128, reverse=True)
        self.conv_up2 = DoubleConv(128, 64, reverse=True)
        self.conv_up1 = Conv3x3BNReLU(64, 64, stride=1)

        self.outconv = nn.Conv2d(64, classes, kernel_size=3, padding=1)

    def forward(self, x):

        # Stage 1
        x1 = self.conv_down1(x)
        x1_size = x1.size()
        x1p, id1 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        x2 = self.conv_down2(x1p)
        x2_size = x2.size()
        x2p, id2 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        x3 = self.conv_down3(x2p)
        x3_size = x3.size()
        x3p, id3 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)

        # Stage 4
        x4 = self.conv_down4(x3p)
        x4_size = x4.size()
        x4p, id4 = F.max_pool2d(x4, kernel_size=2, stride=2, return_indices=True)

        # Stage 5
        x5 = self.conv_down5(x4p)
        x5_size = x5.size()
        x5p, id5 = F.max_pool2d(x5, kernel_size=2, stride=2, return_indices=True)

        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2, output_size=x5_size)
        x5d = self.conv_up5(x5d)

        # Stage 4d
        x4d = F.max_unpool2d(x5d, id4, kernel_size=2, stride=2, output_size=x4_size)
        x4d = self.conv_up4(x4d)

        # Stage 3d
        x3d = F.max_unpool2d(x4d, id3, kernel_size=2, stride=2, output_size=x3_size)
        x3d = self.conv_up3(x3d)

        # Stage 2d
        x2d = F.max_unpool2d(x3d, id2, kernel_size=2, stride=2, output_size=x2_size)
        x2d = self.conv_up2(x2d)

        # Stage 1d
        x1d = F.max_unpool2d(x2d, id1, kernel_size=2, stride=2, output_size=x1_size)
        x1d = self.conv_up1(x1d)

        out = self.outconv(x1d)

        return out



"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegNet(classes=19).to(device)
    summary(model,(3,800,600))
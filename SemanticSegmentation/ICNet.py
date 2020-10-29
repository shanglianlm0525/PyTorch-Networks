# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/10/28 16:41
# @Author : liumin
# @File : ICNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

__all__ = ["ICNet"]


def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

def Conv1x1BNReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


def Conv3x3BN(in_channels,out_channels,stride,dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=dilation,dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels)
        )

def Conv3x3BNReLU(in_channels,out_channels,stride,dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=dilation,dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class CascadeFeatureFusion(nn.Module):
    def __init__(self,low_channels, high_channels, out_channels, num_classes):
        super(CascadeFeatureFusion, self).__init__()

        self.conv_low = Conv3x3BNReLU(low_channels,out_channels,1,dilation=2)
        self.conv_high = Conv3x3BNReLU(high_channels,out_channels,1,dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_low_cls = nn.Conv2d(out_channels, num_classes, 1, bias=False)

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        out = self.relu(x_low + x_high)
        x_low_cls = self.conv_low_cls(x_low)
        return out, x_low_cls


class Backbone(nn.Module):
    def __init__(self, pyramids=[1,2,3,6]):
        super(Backbone, self).__init__()
        self.pretrained = torchvision.models.resnet50(pretrained=True)

    def forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4

class PyramidPoolingModule(nn.Module):
    def __init__(self, pyramids=[1,2,3,6]):
        super(PyramidPoolingModule, self).__init__()
        self.pyramids = pyramids

    def forward(self, x):
        feat = x
        height, width = x.shape[2:]
        for bin_size in self.pyramids:
            feat_x = F.adaptive_avg_pool2d(x, output_size=bin_size)
            feat_x = F.interpolate(feat_x, size=(height, width), mode='bilinear', align_corners=True)
            feat  = feat + feat_x
        return feat


class ICNet(nn.Module):
    def __init__(self, num_classes):
        super(ICNet, self).__init__()

        self.conv_sub1 = nn.Sequential(
            Conv3x3BNReLU(3, 32, 2),
            Conv3x3BNReLU(32, 32, 2),
            Conv3x3BNReLU(32, 64, 2)
        )
        self.backbone = Backbone()
        self.ppm = PyramidPoolingModule()

        self.cff_12 = CascadeFeatureFusion(128, 64, 128, num_classes)
        self.cff_24 = CascadeFeatureFusion(2048, 512, 128, num_classes)

        self.conv_cls = nn.Conv2d(128, num_classes, 1, bias=False)

    def forward(self, x):
        # sub 1
        x_sub1 = self.conv_sub1(x)
        # sub 2
        x_sub2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        _, x_sub2, _, _ = self.backbone(x_sub2)
        # sub 4
        x_sub4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        _, _, _, x_sub4 = self.backbone(x_sub4)

        # add PyramidPoolingModule
        x_sub4 = self.ppm(x_sub4)

        outs = list()
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outs.append(x_24_cls)
        # x_cff_12, x_12_cls = self.cff_12(x_sub2, x_sub1)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)
        outs.append(x_12_cls)

        up_x2 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear')
        up_x2 = self.conv_cls(up_x2)
        outs.append(up_x2)
        up_x8 = F.interpolate(up_x2, scale_factor=4, mode='bilinear')
        outs.append(up_x8)
        # 1 -> 1/4 -> 1/8 -> 1/16
        outs.reverse()

        return outs


if __name__ == '__main__':
    model = ICNet(num_classes=19)
    print(model)

    input = torch.randn(1,3,512,512)
    output = model(input)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(output[3].shape)
# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/31 15:38
# @Author : liumin
# @File : FastSCNN.py

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


def Conv3x3BN(in_channels,out_channels,stride,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,groups=groups),
            nn.BatchNorm2d(out_channels)
        )

def Conv3x3BNReLU(in_channels,out_channels,stride,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

def Conv1x1BNReLU(in_channels,out_channels,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

def Conv1x1BN(in_channels,out_channels,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,groups=groups),
            nn.BatchNorm2d(out_channels)
        )

def DSConv(in_channels, out_channels, stride):
    return nn.Sequential(
        Conv3x3BN(in_channels=in_channels, out_channels=in_channels, stride=stride, groups=in_channels),
        Conv1x1BNReLU(in_channels=in_channels, out_channels=out_channels),
    )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        mid_channels = (in_channels * expansion_factor)

        self.bottleneck = nn.Sequential(
            Conv1x1BNReLU(in_channels, mid_channels),
            Conv3x3BNReLU(mid_channels, mid_channels, stride,groups=mid_channels),
            Conv1x1BN(mid_channels, out_channels)
        )

        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.bottleneck(x)
        out = (out+self.shortcut(x)) if self.stride==1 else out
        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels):
        super(PyramidPooling, self).__init__()
        mid_channels = in_channels // 4
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        self.out = Conv3x3BNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x

class LearningToDownsample(nn.Module):
    def __init__(self):
        super(LearningToDownsample, self).__init__()
        self.conv = Conv3x3BNReLU(in_channels=3, out_channels=32, stride=2)
        self.dsConv1 = DSConv(in_channels=32, out_channels=48, stride=2)
        self.dsConv2 = DSConv(in_channels=48, out_channels=64, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsConv1(x)
        out = self.dsConv2(x)
        return out


class GlobalFeatureExtractor(nn.Module):
    def __init__(self):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(inplanes=64, planes=64, blocks_num=3, stride=2)
        self.bottleneck2 = self._make_layer(inplanes=64, planes=96, blocks_num=3, stride=2)
        self.bottleneck3 = self._make_layer(inplanes=96, planes=128, blocks_num=3, stride=1)
        self.ppm = PyramidPooling(in_channels=128, out_channels=128)

    def _make_layer(self, inplanes, planes, blocks_num, stride=1):
        layers = []
        layers.append(InvertedResidual(inplanes, planes, stride))
        for i in range(1, blocks_num):
            layers.append(InvertedResidual(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        out = self.ppm(x)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, num_classes=20):
        super(FeatureFusionModule, self).__init__()
        self.dsConv1 = nn.Sequential(
            DSConv(in_channels=128, out_channels=128, stride=1),
            Conv3x3BN(in_channels=128, out_channels=128, stride=1)
        )
        self.dsConv2 = DSConv(in_channels=64, out_channels=128, stride=1)

    def forward(self, x, y):
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.dsConv1(x)
        return x + self.dsConv2(y)


class Classifier(nn.Module):
    def __init__(self, num_classes=19):
        super(Classifier, self).__init__()
        self.dsConv = nn.Sequential(
            DSConv(in_channels=128, out_channels=128, stride=1),
            DSConv(in_channels=128, out_channels=128, stride=1)
        )
        self.conv = Conv3x3BNReLU(in_channels=128, out_channels=num_classes, stride=1)

    def forward(self, x):
        x = self.dsConv(x)
        out = self.conv(x)
        return out


class FastSCNN(nn.Module):
    def __init__(self, num_classes):
        super(FastSCNN, self).__init__()
        self.learning_to_downsample = LearningToDownsample()
        self.global_feature_extractor = GlobalFeatureExtractor()
        self.feature_fusion = FeatureFusionModule()
        self.classifier = Classifier(num_classes=num_classes)

    def forward(self, x):
        y = self.learning_to_downsample(x)
        x = self.global_feature_extractor(y)
        x = self.feature_fusion(x,y)
        out = self.classifier(x)
        return out


if __name__ == '__main__':
    model = FastSCNN(num_classes=19)
    print(model)

    input = torch.randn(1,3,1024,2048)
    output = model(input)
    print(output.shape)
# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/1 14:40
# @Author : liumin
# @File : VoVNetV2.py

import torch
import torch.nn as nn
import torchvision

__all__ = ['VoVNet', 'vovnet27_slim', 'vovnet39', 'vovnet57']

def Conv3x3BNReLU(in_channels,out_channels,stride,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


def Conv3x3BN(in_channels,out_channels,stride,groups):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
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

class eSE_Module(nn.Module):
    def __init__(self, channel,ratio = 16):
        super(eSE_Module, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
            )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x)
        z = self.excitation(y)
        return x * z.expand_as(x)

class OSAv2_module(nn.Module):
    def __init__(self, in_channels,mid_channels, out_channels, block_nums=5):
        super(OSAv2_module, self).__init__()

        self._layers = nn.ModuleList()
        self._layers.append(Conv3x3BNReLU(in_channels=in_channels, out_channels=mid_channels, stride=1))
        for idx in range(block_nums-1):
            self._layers.append(Conv3x3BNReLU(in_channels=mid_channels, out_channels=mid_channels, stride=1))


        self.conv1x1 = Conv1x1BNReLU(in_channels+mid_channels*block_nums,out_channels)
        self.ese = eSE_Module(out_channels)
        self.pass_conv1x1 = Conv1x1BNReLU(in_channels, out_channels)

    def forward(self, x):
        residual = x
        outputs = []
        outputs.append(x)
        for _layer in self._layers:
            x = _layer(x)
            outputs.append(x)
        out = self.ese(self.conv1x1(torch.cat(outputs, dim=1)))
        return out + self.pass_conv1x1(residual)


class VoVNet(nn.Module):
    def __init__(self, planes, layers, num_classes=2):
        super(VoVNet, self).__init__()

        self.groups = 1
        self.stage1 = nn.Sequential(
            Conv3x3BNReLU(in_channels=3, out_channels=64, stride=2, groups=self.groups),
            Conv3x3BNReLU(in_channels=64, out_channels=64, stride=1, groups=self.groups),
            Conv3x3BNReLU(in_channels=64, out_channels=128, stride=1, groups=self.groups),
        )

        self.stage2 = self._make_layer(planes[0][0],planes[0][1],planes[0][2],layers[0])

        self.stage3 = self._make_layer(planes[1][0],planes[1][1],planes[1][2],layers[1])

        self.stage4 = self._make_layer(planes[2][0],planes[2][1],planes[2][2],layers[2])

        self.stage5 = self._make_layer(planes[3][0],planes[3][1],planes[3][2],layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=planes[3][2], out_features=num_classes)

    def _make_layer(self, in_channels, mid_channels,out_channels, block_num):
        layers = []
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for idx in range(block_num):
            layers.append(OSAv2_module(in_channels=in_channels, mid_channels=mid_channels, out_channels=out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        out = self.linear(x)
        return out

def vovnet27_slim(**kwargs):
    planes = [[128, 64, 128],
              [128, 80, 256],
              [256, 96, 384],
              [384, 112, 512]]
    layers = [1, 1, 1, 1]
    model = VoVNet(planes, layers)
    return model

def vovnet39(**kwargs):
    planes = [[128, 128, 256],
              [256, 160, 512],
              [512, 192, 768],
              [768, 224, 1024]]
    layers = [1, 1, 2, 2]
    model = VoVNet(planes, layers)
    return model

def vovnet57(**kwargs):
    planes = [[128, 128, 256],
              [256, 160, 512],
              [512, 192, 768],
              [768, 224, 1024]]
    layers = [1, 1, 4, 3]
    model = VoVNet(planes, layers)
    return model


class SAG_Mask(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAG_Mask, self).__init__()
        mid_channels = in_channels

        self.fisrt_convs = nn.Sequential(
            Conv3x3BNReLU(in_channels=in_channels, out_channels=mid_channels, stride=1),
            Conv3x3BNReLU(in_channels=mid_channels, out_channels=mid_channels, stride=1),
            Conv3x3BNReLU(in_channels=mid_channels, out_channels=mid_channels, stride=1),
            Conv3x3BNReLU(in_channels=mid_channels, out_channels=mid_channels, stride=1)
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv3x3 = Conv3x3BNReLU(in_channels=mid_channels*2, out_channels=mid_channels, stride=1)
        self.sigmoid = nn.Sigmoid()

        self.deconv = nn.ConvTranspose2d(mid_channels,mid_channels,kernel_size=2, stride=2)
        self.conv1x1 = Conv1x1BN(mid_channels,out_channels)

    def forward(self, x):
        residual =  x = self.fisrt_convs(x)
        aggregate = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        sag = self.sigmoid(self.conv3x3(aggregate))
        sag_x = residual + sag * x
        out = self.conv1x1(self.deconv(sag_x))
        return out

if __name__=='__main__':
    model = vovnet27_slim()
    print(model)

    input = torch.randn(1, 3, 64, 64)
    out = model(input)
    print(out.shape)

    sag_mask = SAG_Mask(16,80)
    print(sag_mask)
    input = torch.randn(1, 16, 14, 14)
    out = sag_mask(input)
    print(out.shape)

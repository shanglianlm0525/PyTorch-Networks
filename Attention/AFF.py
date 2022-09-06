# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/11/17 10:29
# @Author : liumin
# @File : AFF.py

import torch
import torch.nn as nn


class MS_CAM(nn.Module):
    def __init__(self, channel, ratio = 16):
        super(MS_CAM, self).__init__()
        mid_channel = channel // ratio
        self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=channel, out_channels=mid_channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(mid_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=mid_channel, out_channels=channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channel),
            )

        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=mid_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channel, out_channels=channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        g_x = self.global_att(x)
        l_x = self.local_att(x)
        w = self.sigmoid(l_x * g_x.expand_as(l_x))
        return w * x


class AFF(nn.Module):
    def __init__(self):
        super(AFF, self).__init__()


    def forward(self, x):
        pass


if __name__=='__main__':
    model = MS_CAM(16)
    print(model)

    input = torch.randn(2, 16, 64, 64)
    out = model(input)
    print(out.shape)
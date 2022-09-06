# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/1/17 14:18
# @Author : liumin
# @File : GAM.py

import torch
import torch.nn as nn


class GAM(nn.Module):
    def __init__(self, channels, rate=4):
        super(GAM, self).__init__()
        mid_channels = channels // rate

        self.channel_attention = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # channel attention
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att
        # spatial attention
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
        return out


if __name__ == '__main__':
    x = torch.randn(1, 16, 64, 64)
    b, c, h, w = x.shape
    net = GAM(channels=c)
    out = net(x)
    print(out.shape)
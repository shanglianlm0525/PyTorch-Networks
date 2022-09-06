# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/12/7 11:06
# @Author : liumin
# @File : NAM.py

import torch
import torch.nn as nn

"""
    NAM: Normalization-based Attention Module
    PDF: https://arxiv.org/pdf/2111.12419.pdf
"""

class NAM(nn.Module):
    def __init__(self, channel):
        super(NAM, self).__init__()
        self.channel = channel
        self.bn2 = nn.BatchNorm2d(self.channel, affine=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        out = self.sigmoid(x) * residual  #
        return out


if __name__=='__main__':
    model = NAM(channel=16)
    print(model)

    input = torch.randn(1, 16, 64, 64)
    out = model(input)
    print(out.shape)
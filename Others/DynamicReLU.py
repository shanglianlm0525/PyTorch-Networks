# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/9/11 13:57
# @Author : liumin
# @File : DynamicReLU.py

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class BatchNorm(nn.Module):
    def forward(self, x):
        return 2 * x - 1


class DynamicReLU_A(nn.Module):
    def __init__(self, channels, K=2,ratio=6):
        super(DynamicReLU_A, self).__init__()
        mid_channels = 2*K

        self.K = K
        self.lambdas = torch.Tensor([1.]*K + [0.5]*K).float()
        self.init_v = torch.Tensor([1.] + [0.]*(2*K - 1)).float()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dynamic = nn.Sequential(
            nn.Linear(in_features=channels,out_features=channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channels // ratio, out_features=mid_channels),
            nn.Sigmoid(),
            BatchNorm()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        z = self.dynamic(y)

        relu_coefs = z.view(-1, 2 * self.K) * self.lambdas + self.init_v
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.K] + relu_coefs[:, self.K:]

        output = torch.max(output, dim=-1)[0].transpose(0, -1)
        return output


class DynamicReLU_B(nn.Module):
    def __init__(self, channels, K=2,ratio=6):
        super(DynamicReLU_B, self).__init__()
        mid_channels = 2*K*channels

        self.K = K
        self.channels = channels
        self.lambdas = torch.Tensor([1.]*K + [0.5]*K).float()
        self.init_v = torch.Tensor([1.] + [0.]*(2*K - 1)).float()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dynamic = nn.Sequential(
            nn.Linear(in_features=channels,out_features=channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channels // ratio, out_features=mid_channels),
            nn.Sigmoid(),
            BatchNorm()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        z = self.dynamic(y)

        relu_coefs = z.view(-1, self.channels, 2 * self.K) * self.lambdas + self.init_v
        x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :, :self.K] + relu_coefs[:, :, self.K:]
        output = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)
        return output

if __name__=='__main__':
    model = DynamicReLU_B(64)
    print(model)

    input = torch.randn(1, 64, 56, 56)
    out = model(input)
    print(out.shape)
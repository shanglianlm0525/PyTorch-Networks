# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/28 18:04
# @Author : liumin
# @File : LWnet.py

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

def ConvBNReLU(in_channels,out_channels,kernel_size,stride,padding,dilation=1,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation,groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


def ConvBN(in_channels,out_channels,kernel_size,stride,padding,dilation=1,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation,groups=groups, bias=False),
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

class LWbottleneck(nn.Module):
    def __init__(self, in_channels,out_channels,stride):
        super(LWbottleneck, self).__init__()
        self.stride = stride
        self.pyramid_list = nn.ModuleList()
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=[5,1], stride=stride, padding=[2,0]))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=[1,5], stride=stride, padding=[0,2]))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=[3,1], stride=stride, padding=[1,0]))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=[1,3], stride=stride, padding=[0,1]))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=[2,1], stride=stride, padding=[1,0]))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=[1,2], stride=stride, padding=[0,1]))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=2, stride=stride, padding=1))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=stride, padding=1))

        self.shrink = Conv1x1BN(in_channels*8,out_channels)

    def forward(self, x):
        b,c,w,h = x.shape
        if self.stride>1:
            w, h = w//self.stride,h//self.stride
        outputs = []
        for pyconv in self.pyramid_list:
            pyconv_x = pyconv(x)
            if x.shape[2:] != pyconv_x.shape[2:]:
                pyconv_x = pyconv_x[:,:,:w,:h]
            outputs.append(pyconv_x)
        out = torch.cat(outputs, 1)
        return self.shrink(out)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.stage1 = nn.Sequential(
            ConvBNReLU(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            Conv1x1BN(in_channels=32, out_channels=16),
        )
        self.stage2 = nn.Sequential(
            LWbottleneck(in_channels=16,out_channels=24,stride=2),
            LWbottleneck(in_channels=24, out_channels=24, stride=1),
        )
        self.stage3 = nn.Sequential(
            LWbottleneck(in_channels=24, out_channels=32, stride=2),
            LWbottleneck(in_channels=32, out_channels=32, stride=1),
        )
        self.stage4 = nn.Sequential(
            LWbottleneck(in_channels=32, out_channels=32, stride=2)
        )
        self.stage5 = nn.Sequential(
            LWbottleneck(in_channels=32, out_channels=64, stride=2),
            LWbottleneck(in_channels=64, out_channels=64, stride=1),
            LWbottleneck(in_channels=64, out_channels=64, stride=1),
            LWbottleneck(in_channels=64, out_channels=64, stride=1),
        )

        self.conv1 = Conv1x1BN(in_channels=64, out_channels=320)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = F.pad(x,pad=(0,1,0,1),mode='constant',value=0)
        out1 = x = self.stage3(x)
        x = self.stage4(x)
        x = F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0)
        x = self.stage5(x)
        out2 = self.conv1(x)
        return out1,out2

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.depthwise1 = ConvBNReLU(in_channels, out_channels, 3, 1, 6, dilation=6)
        self.depthwise2 = ConvBNReLU(in_channels, out_channels, 3, 1, 12, dilation=12)
        self.depthwise3 = ConvBNReLU(in_channels, out_channels, 3, 1, 18, dilation=18)
        self.pointconv = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        x1 = self.depthwise1(x)
        x2 = self.depthwise2(x)
        x3 = self.depthwise3(x)
        x4 = self.pointconv(x)
        return torch.cat([x1,x2,x3,x4], dim=1)

class Decoder(nn.Module):
    def __init__(self,num_classes=2):
        super(Decoder, self).__init__()
        self.aspp = ASPP(320, 128)
        self.pconv1 = Conv1x1BN(128*4, 512)

        self.pconv2 = Conv1x1BN(512+32, 128)
        self.pconv3 = Conv1x1BN(128, num_classes)

    def forward(self, x, y):
        x = self.pconv1(self.aspp(x))
        x = F.interpolate(x,y.shape[2:],align_corners=True,mode='bilinear')
        x = torch.cat([x,y], dim=1)
        out = self.pconv3(self.pconv2(x))
        return out

class LW_Network(nn.Module):
    def __init__(self, num_classes=2):
        super(LW_Network, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes)
    def forward(self, x):
        x1,x2 = self.encoder(x)
        out = self.decoder(x2,x1)
        return out



if __name__ == '__main__':
    model = LW_Network()
    print(model)

    input = torch.randn(1, 3, 331, 331)
    output = model(input)
    print(output.shape)
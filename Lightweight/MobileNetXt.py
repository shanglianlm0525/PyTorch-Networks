import torch
import torch.nn as nn
import torchvision
from functools import reduce


def Conv3x3BN(in_channels,out_channels,stride=1,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm2d(out_channels)
        )

def Conv3x3BNReLU(in_channels,out_channels,stride=1,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

def Conv1x1BNReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

class SandglassBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor=6):
        super(SandglassBlock, self).__init__()
        self.stride = stride
        mid_channels = in_channels // expansion_factor
        self.identity = stride == 1 and in_channels == out_channels

        self.bottleneck = nn.Sequential(
            Conv3x3BNReLU(in_channels, in_channels, 1, groups=in_channels),
            Conv1x1BN(in_channels, mid_channels),
            Conv1x1BNReLU(mid_channels, out_channels),
            Conv3x3BN(out_channels, out_channels, stride, groups=out_channels),
        )

    def forward(self, x):
        out = self.bottleneck(x)
        if self.identity:
            return out + x
        else:
            return out


class MobileNetXt(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetXt,self).__init__()

        self.first_conv = Conv3x3BNReLU(3,32,2,groups=1)

        self.layer1 = self.make_layer(in_channels=32, out_channels=96, stride=2, expansion_factor=2, block_num=1)
        self.layer2 = self.make_layer(in_channels=96, out_channels=144, stride=1, expansion_factor=6, block_num=1)
        self.layer3 = self.make_layer(in_channels=144, out_channels=192, stride=2, expansion_factor=6, block_num=3)
        self.layer4 = self.make_layer(in_channels=192, out_channels=288, stride=2, expansion_factor=6, block_num=3)
        self.layer5 = self.make_layer(in_channels=288, out_channels=384, stride=1, expansion_factor=6, block_num=4)
        self.layer6 = self.make_layer(in_channels=384, out_channels=576, stride=2, expansion_factor=6, block_num=4)
        self.layer7 = self.make_layer(in_channels=576, out_channels=960, stride=1, expansion_factor=6, block_num=2)
        self.layer8 = self.make_layer(in_channels=960, out_channels=1280, stride=1, expansion_factor=6, block_num=1)

        self.avgpool = nn.AvgPool2d(kernel_size=7,stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=1280,out_features=num_classes)

    def make_layer(self, in_channels, out_channels, stride, expansion_factor, block_num):
        layers = []
        layers.append(SandglassBlock(in_channels, out_channels, stride,expansion_factor))
        for i in range(1, block_num):
            layers.append(SandglassBlock(out_channels,out_channels,1,expansion_factor))
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.dropout(x)
        out = self.linear(x)
        return out


if __name__=='__main__':
    model = MobileNetXt()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)

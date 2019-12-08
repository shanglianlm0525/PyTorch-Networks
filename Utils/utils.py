import torch
import torch.nn as nn

def Conv3x3BNReLU(in_channels,out_channels,stride,padding=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def Conv1x1BNReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def ConvBNReLU(in_channels,out_channels,kernel_size,stride,padding=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def ConvBN(in_channels,out_channels,kernel_size,stride,padding=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels)
        )

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        mid_channels = out_channels//2

        self.bottleneck = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1),
            ConvBNReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1),
        )
        self.shortcut = ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.bottleneck(x)
        return out+self.shortcut(x)
import torch
import torch.nn as nn
import torchvision

def ConvBNReLU(in_channels,out_channels,kernel_size,stride,padding):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

class ResidualModule(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(ResidualModule, self).__init__()

        self.bottleneck = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0),
            ConvBNReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        out = self.bottleneck(x)
        return out+x


class Hourglass(nn.Module):
    def __init__(self):
        super(Hourglass, self).__init__()

        self.first_conv = ConvBNReLU(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.residual_block = ResidualModule(in_channels=64,  out_channels=256, mid_channels=128)
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)


    def forward(self, x):
        out = self.max_pool(self.residual_block(self.first_conv(x)))

        return out


if __name__ == '__main__':
    model = Hourglass()
    print(model)

    data = torch.randn(1,3,256,256)
    output = model(data)
    print(output.shape)
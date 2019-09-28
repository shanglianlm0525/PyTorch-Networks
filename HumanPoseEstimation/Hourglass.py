import torch
import torch.nn as nn
import torchvision

def ConvBNReLU(in_channels,out_channels,kernel_size,stride,padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
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


class HourglassModule(nn.Module):
    def __init__(self, nChannels=256, nModules=2, numReductions = 4):
        super(HourglassModule, self).__init__()
        self.nChannels = nChannels
        self.nModules = nModules
        self.numReductions = numReductions

        self.residual_block = self._make_residual_layer(self.nModules, self.nChannels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.after_pool_block = self._make_residual_layer(self.nModules, self.nChannels)

        if numReductions > 1:
            self.hourglass_module = HourglassModule(self.nChannels, self.numReductions - 1, self.nModules)
        else:
            self.num1res_block = self._make_residual_layer(self.nModules, self.nChannels)

        self.lowres_block = self._make_residual_layer(self.nModules, self.nChannels)

        self.upsample = nn.Upsample(scale_factor=2)

    def _make_residual_layer(self, nModules, nChannels):
        _residual_blocks = []
        for _ in range(nModules):
            _residual_blocks.append(ResidualBlock(in_channels=nChannels, out_channels=nChannels))
        return nn.Sequential(*_residual_blocks)

    def forward(self, x):
        out1 = self.residual_block(x)

        out2 = self.max_pool(x)
        out2 = self.after_pool_block(out2)

        if self.numReductions > 1:
            out2 = self.hourglass_module(out2)
        else:
            out2 = self.num1res_block(out2)
        out2 = self.lowres_block(out2)
        out2 = self.upsample(out2)

        return out1 + out2

class Hourglass(nn.Module):
    def __init__(self, nJoints):
        super(Hourglass, self).__init__()

        self.first_conv = ConvBNReLU(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.residual_block1 = ResidualBlock(in_channels=64,  out_channels=128)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.residual_block2 = ResidualBlock(in_channels=128, out_channels=128)
        self.residual_block3 = ResidualBlock(in_channels=128, out_channels=256)

        self.hourglass_module1 = HourglassModule(nChannels=256, nModules=2, numReductions = 4)
        self.hourglass_module2 = HourglassModule(nChannels=256, nModules=2, numReductions = 4)

        self.after_hourglass_conv1 = ConvBNReLU(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.proj_conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
        self.out_conv1 = nn.Conv2d(in_channels=256,out_channels=nJoints,kernel_size=1,stride=1)
        self.remap_conv1 = nn.Conv2d(in_channels=nJoints, out_channels=256, kernel_size=1, stride=1)

        self.after_hourglass_conv2 = ConvBNReLU(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.proj_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
        self.out_conv2 = nn.Conv2d(in_channels=256, out_channels=nJoints, kernel_size=1, stride=1)
        self.remap_conv2 = nn.Conv2d(in_channels=nJoints, out_channels=256, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.max_pool(self.residual_block1(self.first_conv(x)))
        x = self.residual_block3(self.residual_block2(x))

        x = self.hourglass_module1(x)
        residual1= x = self.after_hourglass_conv1(x)
        out1 = self.out_conv1(x)
        residual2 =  x = residual1 + self.remap_conv1(out1)+self.proj_conv1(x)

        x = self.hourglass_module2(x)
        x = self.after_hourglass_conv2(x)
        out2 = self.out_conv2(x)
        x = residual2 + self.remap_conv2(out2) + self.proj_conv2(x)

        return out1, out2

if __name__ == '__main__':
    model = Hourglass(nJoints=16)
    print(model)

    data = torch.randn(1,3,256,256)
    out1, out2 = model(data)
    print(out1.shape)
    print(out2.shape)


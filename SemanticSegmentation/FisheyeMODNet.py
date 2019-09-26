import torch
import torch.nn as nn

def Conv3x3BNReLU(in_channels,out_channels,stride,groups):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def Conv1x1BNReLU(in_channels,out_channels,groups):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def Conv1x1BN(in_channels,out_channels,groups):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,groups=groups),
            nn.BatchNorm2d(out_channels)
        )

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


class ShuffleNetUnits(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super(ShuffleNetUnits, self).__init__()
        self.stride = stride
        out_channels = out_channels - in_channels if self.stride>1 else out_channels
        mid_channels = out_channels // 4

        self.bottleneck = nn.Sequential(
            Conv1x1BNReLU(in_channels, mid_channels,groups),
            ChannelShuffle(groups),
            Conv3x3BNReLU(mid_channels, mid_channels, stride,groups),
            Conv1x1BN(mid_channels, out_channels,groups)
        )
        if self.stride>1:
            self.shortcut = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.bottleneck(x)
        out = torch.cat([self.shortcut(x), out], dim=1) if self.stride > 1 else (out + x)
        return self.relu(out)

class FisheyeMODNet(nn.Module):
    def __init__(self, groups=1, num_classes=2):
        super(FisheyeMODNet, self).__init__()
        layers = [4, 8, 4]

        self.stage1a = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3,stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.stage2a = self._make_layer(24, 120, groups, layers[0])

        self.stage1b = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage2b = self._make_layer(24, 120, groups, layers[0])

        self.stage3 = self._make_layer(240, 480, groups, layers[1])
        self.stage4 = self._make_layer(480, 960, groups, layers[2])

        self.adapt_conv3 = nn.Conv2d(960, num_classes, kernel_size=1)
        self.adapt_conv2 = nn.Conv2d(480, num_classes, kernel_size=1)
        self.adapt_conv1 = nn.Conv2d(240, num_classes, kernel_size=1)

        self.up_sampling3 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2, padding=1)
        self.up_sampling2 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2, padding=1)
        self.up_sampling1 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=16, stride=8, padding=4)

        self.softmax  = nn.Softmax(dim=1)

        self.init_params()

    def _make_layer(self, in_channels, out_channels, groups, block_num):
        layers = []
        layers.append(ShuffleNetUnits(in_channels=in_channels, out_channels=out_channels, stride=2, groups=groups))
        for idx in range(1, block_num):
            layers.append(ShuffleNetUnits(in_channels=out_channels, out_channels=out_channels, stride=1, groups=groups))
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        x = self.stage2a(self.stage1a(x))
        y = self.stage2b(self.stage1b(y))
        feature1 = torch.cat([x, y], dim=1)
        feature2 = self.stage3(feature1)
        feature3 = self.stage4(feature2)

        out3 = self.up_sampling3(self.adapt_conv3(feature3))
        out2 = self.up_sampling2(self.adapt_conv2(feature2) + out3)
        out1 = self.up_sampling1(self.adapt_conv1(feature1) + out2)

        out = self.softmax(out1)
        return out


if __name__ == '__main__':
    model = FisheyeMODNet()

    input1 = torch.randn(1, 3, 640, 640)
    input2 = torch.randn(1, 3, 640, 640)

    out = model(input1, input2)
    print(out.shape)
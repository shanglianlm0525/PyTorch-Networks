import torch
import torch.nn as nn
import torchvision

def Conv3x3BNReLU(in_channels,out_channels,stride,groups):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def Conv3x3BN(in_channels,out_channels,stride,groups):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,groups=groups),
            nn.BatchNorm2d(out_channels)
        )

def Conv1x1BNReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

class HalfSplit(nn.Module):
    def __init__(self, dim=0, first_half=True):
        super(HalfSplit, self).__init__()
        self.first_half = first_half
        self.dim = dim

    def forward(self, input):
        splits = torch.chunk(input, 2, dim=self.dim)
        return splits[0] if self.first_half else splits[1]

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
        if self.stride > 1:
            mid_channels = out_channels - in_channels
        else:
            mid_channels = out_channels // 2
            in_channels = mid_channels
            self.first_half = HalfSplit(dim=1, first_half=True)
            self.second_split = HalfSplit(dim=1, first_half=False)

        self.bottleneck = nn.Sequential(
            Conv1x1BNReLU(in_channels, in_channels),
            Conv3x3BN(in_channels, mid_channels, stride, groups),
            Conv1x1BNReLU(mid_channels, mid_channels)
        )

        if self.stride > 1:
            self.shortcut = nn.Sequential(
                Conv3x3BN(in_channels=in_channels, out_channels=in_channels, stride=stride, groups=groups),
                Conv1x1BNReLU(in_channels, in_channels)
            )

        self.channel_shuffle = ChannelShuffle(groups)

    def forward(self, x):
        if self.stride > 1:
            x1 = self.bottleneck(x)
            x2 = self.shortcut(x)
        else:
            x1 = self.first_half(x)
            x2 = self.second_split(x)
            x1 = self.bottleneck(x1)

        out = torch.cat([x1, x2], dim=1)
        out = self.channel_shuffle(out)
        return out

class ShuffleNetV2(nn.Module):
    def __init__(self, planes, layers, groups, num_classes=1000):
        super(ShuffleNetV2, self).__init__()
        self.groups = groups
        self.stage1 = nn.Sequential(
            Conv3x3BNReLU(in_channels=3, out_channels=24, stride=2, groups=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage2 = self._make_layer(24, planes[0], layers[0], True)
        self.stage3 = self._make_layer(planes[0], planes[1], layers[1], False)
        self.stage4 = self._make_layer(planes[1], planes[2], layers[2], False)

        self.global_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=planes[2] * 7 * 7, out_features=num_classes)

        self.init_params()

    def _make_layer(self, in_channels, out_channels, block_num, is_stage2):
        layers = []
        layers.append(ShuffleNetUnits(in_channels=in_channels, out_channels=out_channels, stride= 2, groups=1 if is_stage2 else self.groups))
        for idx in range(1, block_num):
            layers.append(ShuffleNetUnits(in_channels=out_channels, out_channels=out_channels, stride=1, groups=self.groups))
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
#
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.linear(x)
        return out

def shufflenet_v2_x2_0(**kwargs):
    planes = [244, 488, 976]
    layers = [4, 8, 4]
    model = ShuffleNetV2(planes, layers, 1)
    return model

def shufflenet_v2_x1_5(**kwargs):
    planes = [176, 352, 704]
    layers = [4, 8, 4]
    model = ShuffleNetV2(planes, layers, 1)
    return model

def shufflenet_v2_x1_0(**kwargs):
    planes = [116, 232, 464]
    layers = [4, 8, 4]
    model = ShuffleNetV2(planes, layers, 1)
    return model

def shufflenet_v2_x0_5(**kwargs):
    planes = [48, 96, 192]
    layers = [4, 8, 4]
    model = ShuffleNetV2(planes, layers, 1)
    return model

if __name__ == '__main__':
    model = shufflenet_v2_x1_0()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
import torch
import torch.nn as nn
import torchvision

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
    def __init__(self, in_channels, out_channels, stride,groups):
        super(ShuffleNetUnits, self).__init__()
        self.stride = stride
        out_channels = out_channels - in_channels if self.stride >1 else out_channels
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
        out = torch.cat([self.shortcut(x),out],dim=1) if self.stride >1 else (out + x)
        return self.relu(out)

class ShuffleNet(nn.Module):
    def __init__(self, planes, layers, groups, num_classes=1000):
        super(ShuffleNet, self).__init__()

        self.stage1 = nn.Sequential(
            Conv3x3BNReLU(in_channels=3,out_channels=24,stride=2, groups=1),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.stage2 = self._make_layer(24,planes[0], groups, layers[0], True)
        self.stage3 = self._make_layer(planes[0],planes[1], groups, layers[1], False)
        self.stage4 = self._make_layer(planes[1],planes[2], groups, layers[2], False)

        self.global_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=planes[2]*7*7, out_features=num_classes)

        self.init_params()

    def _make_layer(self, in_channels,out_channels, groups, block_num, is_stage2):
        layers = []
        layers.append(ShuffleNetUnits(in_channels=in_channels, out_channels=out_channels, stride=2, groups=1 if is_stage2 else groups))
        for idx in range(1, block_num):
            layers.append(ShuffleNetUnits(in_channels=out_channels, out_channels=out_channels, stride=1, groups=groups))
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.linear(x)
        return out

def shufflenet_g8(**kwargs):
    planes = [384, 768, 1536]
    layers = [4, 8, 4]
    model = ShuffleNet(planes, layers, groups=8)
    return model

def shufflenet_g4(**kwargs):
    planes = [272, 544, 1088]
    layers = [4, 8, 4]
    model = ShuffleNet(planes, layers, groups=4)
    return model

def shufflenet_g3(**kwargs):
    planes = [240, 480, 960]
    layers = [4, 8, 4]
    model = ShuffleNet(planes, layers, groups=3)
    return model

def shufflenet_g2(**kwargs):
    planes = [200, 400, 800]
    layers = [4, 8, 4]
    model = ShuffleNet(planes, layers, groups=2)
    return model

def shufflenet_g1(**kwargs):
    planes = [144, 288, 576]
    layers = [4, 8, 4]
    model = ShuffleNet(planes, layers, groups=1)
    return model

if __name__ == '__main__':
    model = shufflenet_g1()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
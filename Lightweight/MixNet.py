import torch
import torch.nn as nn

class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return x*self.relu6(x+3)/6

def ConvBNActivation(in_channels,out_channels,kernel_size,stride,activate):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish(inplace=True)
        )

def Conv1x1BNActivation(in_channels,out_channels,activate):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish(inplace=True)
        )

def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

class MDConv(nn.Module):
    def __init__(self, nchannels, kernel_sizes, stride):
        super(MDConv,self).__init__()
        self.nchannels = nchannels
        self.groups = len(kernel_sizes)

        self.split_channels = [nchannels // self.groups for _ in range(self.groups)]
        self.split_channels[0] += nchannels - sum(self.split_channels)

        self.layers = []
        for i in range(self.groups):
            self.layers.append(nn.Conv2d(in_channels=self.split_channels[i],out_channels=self.split_channels[i],
                                         kernel_size=kernel_sizes[i], stride=stride,padding=int(kernel_sizes[i]//2), groups=self.split_channels[i]))

    def forward(self, x):
        split_x = torch.split(x, self.split_channels, dim=1)
        outputs = [layer(sp_x) for layer,sp_x in zip(self.layers, split_x)]
        return torch.cat(outputs, dim=1)

class SqueezeAndExcite(nn.Module):
    def __init__(self, nchannels, squeeze_channels, se_ratio=1):
        super(SqueezeAndExcite, self).__init__()
        squeeze_channels = int(squeeze_channels * se_ratio)

        self.SEblock = nn.Sequential(
            nn.Conv2d(in_channels=nchannels, out_channels=squeeze_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=squeeze_channels, out_channels=nchannels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = torch.mean(x, (2, 3), keepdim=True)
        out = self.SEblock(out)
        return out * x

class MDConvBlock(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_sizes, stride,expand_ratio, activate='relu', se_ratio=1):
        super(MDConvBlock,self).__init__()
        self.stride = stride
        self.se_ratio = se_ratio
        mid_channels = in_channels * expand_ratio

        self.expand_conv = Conv1x1BNActivation(in_channels, mid_channels, activate)
        self.md_conv = nn.Sequential(
            # in_channels,out_channels,groups, kernel_sizes, strides
            MDConv(mid_channels, kernel_sizes, stride),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish(inplace=True),
        )
        if self.se_ratio > 0:
            self.squeeze_excite =  SqueezeAndExcite(mid_channels, in_channels)

        self.projection_conv = Conv1x1BN(mid_channels,out_channels)

    def forward(self, x):
        x = self.expand_conv(x)
        x = self.md_conv(x)
        if self.se_ratio > 0:
            x = self.squeeze_excite(x)
        out = self.projection_conv(x)
        return out

class MixNet(nn.Module):
    mixnet_s = [(16, 16, [3], 1, 1, 'ReLU', 0.0),
                (16, 24, [3], 2, 6, 'ReLU', 0.0),
                (24, 24, [3], 1, 3, 'ReLU', 0.0),
                (24, 40, [3, 5, 7], 2, 6, 'Swish', 0.5),
                (40, 40, [3, 5], 1, 6, 'Swish', 0.5),
                (40, 40, [3, 5], 1, 6, 'Swish', 0.5),
                (40, 40, [3, 5], 1, 6, 'Swish', 0.5),
                (40, 80, [3, 5, 7], 2, 6, 'Swish', 0.25),
                (80, 80, [3, 5], 1, 6, 'Swish', 0.25),
                (80, 80, [3, 5], 1, 6, 'Swish', 0.25),
                (80, 120, [3, 5, 7], 1, 6, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5),
                (120, 200, [3, 5, 7, 9, 11], 2, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5)
                ]

    mixnet_m = [(24, 24, [3], 1, 1, 'ReLU', 0.0),
                (24, 32, [3, 5, 7], 2, 6, 'ReLU', 0.0),
                (32, 32, [3], 1, 3, 'ReLU', 0.0),
                (32, 40, [3, 5, 7, 9], 2, 6, 'Swish', 0.5),
                (40, 40, [3, 5], 1, 6, 'Swish', 0.5),
                (40, 40, [3, 5], 1, 6, 'Swish', 0.5),
                (40, 40, [3, 5], 1, 6, 'Swish', 0.5),
                (40, 80, [3, 5, 7], 2, 6, 'Swish', 0.25),
                (80, 80, [3, 5, 7, 9], 1, 6, 'Swish', 0.25),
                (80, 80, [3, 5, 7, 9], 1, 6, 'Swish', 0.25),
                (80, 80, [3, 5, 7, 9], 1, 6, 'Swish', 0.25),
                (80, 120, [3], 1, 6, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5),
                (120, 200, [3, 5, 7, 9], 2, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5)]

    def __init__(self, type='mixnet_s'):
        super(MixNet,self).__init__()

        if type == 'mixnet_s':
            config = self.mixnet_s
            stem_channels = 16
        elif type == 'mixnet_m':
            config = self.mixnet_m
            stem_channels = 24

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=stem_channels,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(stem_channels),
            HardSwish(inplace=True),
        )

        layers = []
        for in_channels, out_channels, kernel_sizes, stride, expand_ratio, activate, se_ratio in config:
            layers.append(MDConvBlock(
                in_channels,
                out_channels,
                kernel_sizes=kernel_sizes,
                stride=stride,
                expand_ratio=expand_ratio,
                activate=activate,
                se_ratio=se_ratio
            ))
        self.bottleneck = nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        out = self.bottleneck(x)
        return out

if __name__ == '__main__':
    model = MixNet(type ='mixnet_m')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
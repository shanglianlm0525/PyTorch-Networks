import torch
import torch.nn as nn
import torchvision

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
    def __init__(self, in_channels,out_channels,groups, kernel_sizes, strides):
        super(MDConv,self).__init__()
        self.in_channels = in_channels
        self.groups = groups

        self.layers = []
        for i in range(groups):
            self.layers.append(nn.Conv2d(in_channels=in_channels//groups,out_channels=out_channels//groups,kernel_size=kernel_sizes[i], stride=strides[i],padding=(kernel_sizes[i]-1)//2))

    def forward(self, x):
        split_x = torch.split(x,self.in_channels//self.groups, dim=1)
        outputs = [layer(sp_x) for layer,sp_x in zip(self.layers, split_x)]
        return torch.cat(outputs, dim=1)

class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channels, out_channels,se_kernel_size, divide=4):
        super(SqueezeAndExcite, self).__init__()
        mid_channels = in_channels // divide
        self.pool = nn.AvgPool2d(kernel_size=se_kernel_size,stride=1)
        self.SEblock = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=mid_channels),
            nn.ReLU6(inplace=True),
            nn.Linear(in_features=mid_channels, out_features=out_channels),
            HardSwish(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.pool(x)
        out = out.view(b, -1)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x

class MDConvBlock(nn.Module):
    def __init__(self, in_channels,mid_channels,out_channels,groups, kernel_sizes, strides, activate='relu', use_se=False,se_kernel_size=1):
        super(MDConvBlock,self).__init__()
        self.stride = strides
        self.use_se = use_se

        self.expand_conv = Conv1x1BNActivation(in_channels, mid_channels, activate)
        self.md_conv = nn.Sequential(
            # in_channels,out_channels,groups, kernel_sizes, strides
            MDConv(mid_channels,mid_channels,groups, kernel_sizes, strides),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish(inplace=True),
        )
        if self.use_se:
            self.squeeze_excite =  SqueezeAndExcite(mid_channels, mid_channels,se_kernel_size)

        self.projection_conv = Conv1x1BN(mid_channels,out_channels)

    def forward(self, x):
        x = self.expand_conv(x)
        x = self.md_conv(x)
        if self.use_se:
            x = self.squeeze_excite(x)
        out = self.projection_conv(x)
        return out

class MixNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(MixNet,self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(16),
            HardSwish(inplace=True),
        )

        self.bottleneck1 = nn.Sequential(
            MDConvBlock(in_channels=16, mid_channels=16, out_channels=16, groups=1,  kernel_sizes=[3], strides=[1],activate='relu', use_se=False,se_kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            MDConvBlock(in_channels=16, mid_channels=16, out_channels=24, groups=1, kernel_sizes=[3], strides=[2],activate='relu', use_se=False,se_kernel_size=1),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True),
            MDConvBlock(in_channels=24, mid_channels=24, out_channels=24, groups=1, kernel_sizes=[3], strides=[1],activate='relu', use_se=False,se_kernel_size=1),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True),
            MDConvBlock(in_channels=24, mid_channels=24, out_channels=40, groups=3, kernel_sizes=[3, 5, 7],strides=[2, 2, 2], activate='hswish', use_se=False, se_kernel_size=1),
            nn.BatchNorm2d(40),
            nn.ReLU6(inplace=True),
        )

        self.bottleneck2 = nn.Sequential(
            MDConvBlock(in_channels=16, mid_channels=1, out_channels=16,groups=1, kernel_sizes=[3], strides=[1], activate='hswish', use_se=False,se_kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            MDConvBlock(in_channels=16, mid_channels=1, out_channels=24,groups=1, kernel_sizes=[3], strides=[2], activate='hswish', use_se=False,se_kernel_size=1),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True),
            MDConvBlock(in_channels=24, mid_channels=1, out_channels=24,groups=1, kernel_sizes=[3], strides=[1], activate='hswish', use_se=False,se_kernel_size=1),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True),
        )

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
        x = self.bottleneck1(x)
        out = self.bottleneck2(x)
        return out


def MixNet_S():
    planes = [200, 400, 800]
    layers = [4, 8, 4]
    model = MixNet()
    return model

def MixNet_M():
    planes = [200, 400, 800]
    layers = [4, 8, 4]
    model = MixNet(planes, layers, groups=2)
    return model

if __name__ == '__main__':
    model = MixNet_S()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
import torch
import torch.nn as nn
import torchvision


class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channels, out_channels, divide=4):
        super(SqueezeAndExcite, self).__init__()
        mid_channels = in_channels // divide
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.SEblock = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=mid_channels),
            nn.ReLU6(inplace=True),
            nn.Linear(in_features=mid_channels, out_features=out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.pool(x)
        out = out.view(b, -1)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x


def Conv1x1BNReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def Conv3x3BNReLU(in_channels,out_channels,stride):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


def VarGConv(in_channels,out_channels,kernel_size,stride,S):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, groups=in_channels // S,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.PReLU(),
    )

def VarGPointConv(in_channels, out_channels,stride,S,isRelu):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride, padding=0, groups=in_channels // S,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.PReLU() if isRelu else nn.Sequential(),
    )

class VarGBlock_S1(nn.Module):
    def __init__(self, in_plances,kernel_size, stride=1, S=8):
        super(VarGBlock_S1, self).__init__()
        plances = 2 * in_plances
        self.varGConv1 = VarGConv(in_plances, plances, kernel_size, stride, S)
        self.varGPointConv1 = VarGPointConv(plances, in_plances, stride, S, isRelu=True)
        self.varGConv2 = VarGConv(in_plances, plances, kernel_size, stride, S)
        self.varGPointConv2 = VarGPointConv(plances, in_plances, stride, S, isRelu=False)
        self.se =  SqueezeAndExcite(in_plances,in_plances)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = x
        x = self.varGPointConv1(self.varGConv1(x))
        x = self.varGPointConv2(self.varGConv2(x))
        x = self.se(x)
        out += x
        return self.prelu(out)

class VarGBlock_S2(nn.Module):
    def __init__(self, in_plances,kernel_size, stride=2, S=8):
        super(VarGBlock_S2, self).__init__()
        plances = 2 * in_plances

        self.varGConvBlock_branch1 = nn.Sequential(
            VarGConv(in_plances, plances, kernel_size, stride, S),
            VarGPointConv(plances, plances, 1, S, isRelu=True),
        )
        self.varGConvBlock_branch2 = nn.Sequential(
            VarGConv(in_plances, plances, kernel_size, stride, S),
            VarGPointConv(plances, plances, 1, S, isRelu=True),
        )

        self.varGConvBlock_3 = nn.Sequential(
            VarGConv(plances, plances*2, kernel_size, 1, S),
            VarGPointConv(plances*2, plances, 1, S, isRelu=False),
        )
        self.shortcut = nn.Sequential(
            VarGConv(in_plances, plances, kernel_size, stride, S),
            VarGPointConv(plances, plances, 1, S, isRelu=False),
        )
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.shortcut(x)
        x1 = x2 = x
        x1= self.varGConvBlock_branch1(x1)
        x2 = self.varGConvBlock_branch2(x2)
        x_new = x1 + x2
        x_new = self.varGConvBlock_3(x_new)
        out += x_new
        return self.prelu(out)


class HeadBlock(nn.Module):
    def __init__(self, in_plances, kernel_size, S=8):
        super(HeadBlock, self).__init__()

        self.varGConvBlock = nn.Sequential(
            VarGConv(in_plances, in_plances, kernel_size, 2, S),
            VarGPointConv(in_plances, in_plances, 1, S, isRelu=True),
            VarGConv(in_plances, in_plances, kernel_size, 1, S),
            VarGPointConv(in_plances, in_plances, 1, S, isRelu=False),
         )

        self.shortcut = nn.Sequential(
            VarGConv(in_plances, in_plances, kernel_size, 2, S),
            VarGPointConv(in_plances, in_plances, 1, S, isRelu=False),
        )

    def forward(self, x):
        out = self.shortcut(x)
        x = self.varGConvBlock(x)
        out += x
        return out


class TailEmbedding(nn.Module):
    def __init__(self, in_plances, plances=512, S=8):
        super(TailEmbedding, self).__init__()
        self.embedding = nn.Sequential(
            Conv1x1BNReLU(in_plances, 1024),
            nn.Conv2d(1024, 1024, 7, 1, padding=0, groups=1024 // S,
                      bias=False),
            nn.Conv2d(1024, 512, 1, 1, padding=0, groups=512, bias=False),
        )

        self.fc = nn.Linear(in_features=512,out_features=plances)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0),-1)
        out = self.fc(x)
        return out


class VarGFaceNet(nn.Module):
    def __init__(self, num_classes=512):
        super(VarGFaceNet, self).__init__()
        S = 8

        self.conv1 = Conv3x3BNReLU(3, 40, 1)
        self.head = HeadBlock(40,3)
        self.stage2 = nn.Sequential(
            VarGBlock_S2(40,3,2),
            VarGBlock_S1(80, 3, 1),
            VarGBlock_S1(80, 3, 1),
        )
        self.stage3 = nn.Sequential(
            VarGBlock_S2(80, 3, 2),
            VarGBlock_S1(160, 3, 1),
            VarGBlock_S1(160, 3, 1),
            VarGBlock_S1(160, 3, 1),
            VarGBlock_S1(160, 3, 1),
            VarGBlock_S1(160, 3, 1),
            VarGBlock_S1(160, 3, 1),
        )
        self.stage4 = nn.Sequential(
            VarGBlock_S2(160, 3, 2),
            VarGBlock_S1(320, 3, 1),
            VarGBlock_S1(320, 3, 1),
            VarGBlock_S1(320, 3, 1),
        )

        self.tail = TailEmbedding(320,num_classes)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.head(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        out= self.tail(x)
        return out


if __name__ == '__main__':
    model = VarGFaceNet()
    print(model)

    input = torch.randn(1, 3, 112, 112)
    out = model(input)
    print(out.shape)
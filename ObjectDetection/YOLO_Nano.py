import torch
import torch.nn as nn
import torchvision

def Conv3x3BNReLU(in_channels,out_channels,stride):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def DWConv3x3BNReLU(in_channels,out_channels,stride):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

def Conv1x1BNReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

class EP(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor=6):
        super(EP, self).__init__()
        self.stride = stride
        mid_channels = (in_channels * expansion_factor)

        self.bottleneck = nn.Sequential(
            Conv1x1BNReLU(in_channels, mid_channels),
            DWConv3x3BNReLU(mid_channels, mid_channels, stride),
            Conv1x1BN(mid_channels, out_channels)
        )

        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.bottleneck(x)
        out = (out + self.shortcut(x)) if self.stride == 1 else out
        return out


class PEP(nn.Module):
    def __init__(self, in_channels, proj_channels, out_channels, stride, expansion_factor=6):
        super(PEP, self).__init__()
        self.stride = stride
        mid_channels = (proj_channels * expansion_factor)

        self.bottleneck = nn.Sequential(
            Conv1x1BNReLU(in_channels, proj_channels),
            Conv1x1BNReLU(proj_channels, mid_channels),
            DWConv3x3BNReLU(mid_channels, mid_channels, stride),
            Conv1x1BN(mid_channels, out_channels)
        )

        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.bottleneck(x)
        out = (out + self.shortcut(x)) if self.stride == 1 else out
        return out


class FCA(nn.Module):
    def __init__(self, channel,ratio = 8):
        super(FCA, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel // ratio, out_features=channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class YOLO_Nano(nn.Module):
    def __init__(self, out_channel=75):
        super(YOLO_Nano, self).__init__()

        self.stage1 = nn.Sequential(
            Conv3x3BNReLU(in_channels=3, out_channels=12, stride=1),
            Conv3x3BNReLU(in_channels=12, out_channels=24, stride=2),
            PEP(in_channels=24, proj_channels=7, out_channels=24, stride=1),
            EP(in_channels=24, out_channels=70, stride=2),
            PEP(in_channels=70, proj_channels=25, out_channels=70, stride=1),
            PEP(in_channels=70, proj_channels=24, out_channels=70, stride=1),
            EP(in_channels=70, out_channels=150, stride=2),
            PEP(in_channels=150, proj_channels=56, out_channels=150, stride=1),
            Conv1x1BNReLU(in_channels=150, out_channels=150),
            FCA(channel=150,ratio=8),
            PEP(in_channels=150, proj_channels=73, out_channels=150, stride=1),
            PEP(in_channels=150, proj_channels=71, out_channels=150, stride=1),
            PEP(in_channels=150, proj_channels=75, out_channels=150, stride=1),
        )

        self.stage2 = nn.Sequential(
            EP(in_channels=150, out_channels=325, stride=2),
        )

        self.stage3 = nn.Sequential(
            PEP(in_channels=325, proj_channels=132, out_channels=325, stride=1),
            PEP(in_channels=325, proj_channels=124, out_channels=325, stride=1),
            PEP(in_channels=325, proj_channels=141, out_channels=325, stride=1),
            PEP(in_channels=325, proj_channels=140, out_channels=325, stride=1),
            PEP(in_channels=325, proj_channels=137, out_channels=325, stride=1),
            PEP(in_channels=325, proj_channels=135, out_channels=325, stride=1),
            PEP(in_channels=325, proj_channels=133, out_channels=325, stride=1),
            PEP(in_channels=325, proj_channels=140, out_channels=325, stride=1),
        )

        self.stage4 = nn.Sequential(
            EP(in_channels=325, out_channels=545, stride=2),
            PEP(in_channels=545, proj_channels=276, out_channels=545, stride=1),
            Conv1x1BNReLU(in_channels=545, out_channels=230),
            EP(in_channels=230, out_channels=489, stride=1),
            PEP(in_channels=489, proj_channels=213, out_channels=469, stride=1),
            Conv1x1BNReLU(in_channels=469, out_channels=189),
        )

        self.stage5 = nn.Sequential(
            Conv1x1BNReLU(in_channels=189, out_channels=105),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.stage6 = nn.Sequential(
            PEP(in_channels=105+325, proj_channels=113, out_channels=325, stride=1),
            PEP(in_channels=325, proj_channels=99, out_channels=207, stride=1),
            Conv1x1BNReLU(in_channels=207, out_channels=98),
        )

        self.stage7 = nn.Sequential(
            Conv1x1BNReLU(in_channels=98, out_channels=47),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.out_stage1 = nn.Sequential(
            PEP(in_channels=150+47, proj_channels=58, out_channels=122, stride=1),
            PEP(in_channels=122, proj_channels=52, out_channels=87, stride=1),
            PEP(in_channels=87, proj_channels=47, out_channels=93, stride=1),
            Conv1x1BNReLU(in_channels=93, out_channels=out_channel),
        )

        self.out_stage2 = nn.Sequential(
            EP(in_channels=98, out_channels=183, stride=1),
            Conv1x1BNReLU(in_channels=183, out_channels=out_channel),
        )

        self.out_stage3 = nn.Sequential(
            EP(in_channels=189, out_channels=462, stride=1),
            Conv1x1BNReLU(in_channels=462, out_channels=out_channel),
        )

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        x6 = self.stage6(torch.cat([x3,x5], dim=1))
        x7 = self.stage7(x6)
        out1 = self.out_stage1(torch.cat([x1,x7], dim=1))
        out2 = self.out_stage2(x6)
        out3 = self.out_stage3(x4)
        return out1, out2, out3

if __name__ == '__main__':
    model = YOLO_Nano()
    print(model)

    input = torch.randn(1,3,416,416)
    output1,output2,output3  = model(input)
    print(output1.shape)
    print(output2.shape)
    print(output3.shape)
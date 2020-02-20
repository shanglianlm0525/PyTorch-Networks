import torch
import torch.nn as nn
import torchvision


class cSE_Module(nn.Module):
    def __init__(self, channel,ratio = 16):
        super(cSE_Module, self).__init__()
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


class sSE_Module(nn.Module):
    def __init__(self, channel):
        super(sSE_Module, self).__init__()
        self.spatial_excitation = nn.Sequential(
                nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1,stride=1,padding=0),
                nn.Sigmoid()
            )
    def forward(self, x):
        z = self.spatial_excitation(x)
        return x * z.expand_as(x)


class scSE_Module(nn.Module):
    def __init__(self, channel,ratio = 16):
        super(scSE_Module, self).__init__()
        self.cSE = cSE_Module(channel,ratio)
        self.sSE = sSE_Module(channel)

    def forward(self, x):
        return self.cSE(x) + self.sSE(x)


if __name__=='__main__':
    # model = cSE_Module(channel=16)
    # model = sSE_Module(channel=16)
    model = scSE_Module(channel=16)
    print(model)

    input = torch.randn(1, 16, 64, 64)
    out = model(input)
    print(out.shape)
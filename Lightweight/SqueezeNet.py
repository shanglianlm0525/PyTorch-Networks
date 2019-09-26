import torch
import torch.nn as nn
import torchvision

class FireModule(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(FireModule, self).__init__()
        mid_channels = out_channels//4

        self.squeeze = nn.Conv2d(in_channels=in_channels,out_channels=mid_channels,kernel_size=1,stride=1)
        self.squeeze_relu = nn.ReLU6(inplace=True)

        self.expand3x3 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1,padding=1)
        self.expand3x3_relu = nn.ReLU6(inplace=True)

        self.expand1x1 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.expand1x1_relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.squeeze_relu(self.squeeze(x))
        y = self.expand3x3_relu(self.expand3x3(x))
        z = self.expand1x1_relu(self.expand1x1(x))
        out = torch.cat([y, z],dim=1)
        return out

class SqueezeNet(nn.Module):
    def __init__(self, num_classes = 1000):
        super(SqueezeNet, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(96),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),

            FireModule(in_channels=96, out_channels=64),
            FireModule(in_channels=128, out_channels=64),
            FireModule(in_channels=128, out_channels=128),
            nn.MaxPool2d(kernel_size=3,stride=2),

            FireModule(in_channels=256, out_channels=128),
            FireModule(in_channels=256, out_channels=192),
            FireModule(in_channels=384, out_channels=192),
            FireModule(in_channels=384, out_channels=256),
            nn.MaxPool2d(kernel_size=3, stride=2),

            FireModule(in_channels=512, out_channels=256),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=13, stride=1),
        )

    def forward(self, x):
        out = self.bottleneck(x)
        return out.view(out.size(1),-1)

if __name__ == '__main__':
    model = SqueezeNet()
    print(model)

    input = torch.rand(1,3,224,224)
    out = model(input)
    print(out.shape)


import torch
import torch.nn as nn

def Conv3x3BNReLU(in_channels,out_channels,padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

def Conv1x1BNReLU(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

class Darknet19(nn.Module):
    def __init__(self, num_classes=1000):
        super(Darknet19, self).__init__()

        self.feature = nn.Sequential(
            Conv3x3BNReLU(in_channels=3, out_channels=32),
            nn.MaxPool2d(kernel_size=2,stride=2),
            Conv3x3BNReLU(in_channels=32, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv3x3BNReLU(in_channels=64, out_channels=128),
            Conv1x1BNReLU(in_channels=128, out_channels=64),
            Conv3x3BNReLU(in_channels=64, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv3x3BNReLU(in_channels=128, out_channels=256),
            Conv1x1BNReLU(in_channels=256, out_channels=128),
            Conv3x3BNReLU(in_channels=128, out_channels=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv3x3BNReLU(in_channels=256, out_channels=512),
            Conv1x1BNReLU(in_channels=512, out_channels=256),
            Conv3x3BNReLU(in_channels=256, out_channels=512),
            Conv1x1BNReLU(in_channels=512, out_channels=256),
            Conv3x3BNReLU(in_channels=256, out_channels=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv3x3BNReLU(in_channels=512, out_channels=1024),
            Conv1x1BNReLU(in_channels=1024, out_channels=512),
            Conv3x3BNReLU(in_channels=512, out_channels=1024),
            Conv1x1BNReLU(in_channels=1024, out_channels=512),
            Conv3x3BNReLU(in_channels=512, out_channels=1024),
        )

        self.classifier = nn.Sequential(
            Conv1x1BNReLU(in_channels=1024, out_channels=num_classes),
            nn.AvgPool2d(kernel_size=7,stride=1),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        x = torch.squeeze(x, dim=3).contiguous()
        x = torch.squeeze(x, dim=2).contiguous()
        out = self.softmax(x)
        return out

if __name__ == '__main__':
    model = Darknet19()
    print(model)

    input = torch.randn(1,3,224,224)
    out = model(input)
    print(out.shape)
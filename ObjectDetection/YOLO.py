import torch
import torch.nn as nn

def Conv1x1BNReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def Conv3x3BNReLU(in_channels,out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64, kernel_size=7,stride=2,padding=3),
            nn.MaxPool2d(kernel_size=2,stride=2),
            Conv3x3BNReLU(in_channels=64, out_channels=192),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv1x1BNReLU(in_channels=192, out_channels=128),
            Conv3x3BNReLU(in_channels=128, out_channels=256),
            Conv1x1BNReLU(in_channels=256, out_channels=256),
            Conv3x3BNReLU(in_channels=256, out_channels=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv1x1BNReLU(in_channels=512, out_channels=256),
            Conv3x3BNReLU(in_channels=256, out_channels=512),
            Conv1x1BNReLU(in_channels=512, out_channels=256),
            Conv3x3BNReLU(in_channels=256, out_channels=512),
            Conv1x1BNReLU(in_channels=512, out_channels=256),
            Conv3x3BNReLU(in_channels=256, out_channels=512),
            Conv1x1BNReLU(in_channels=512, out_channels=256),
            Conv3x3BNReLU(in_channels=256, out_channels=512),
            Conv1x1BNReLU(in_channels=512, out_channels=512),
            Conv3x3BNReLU(in_channels=512, out_channels=1024),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv1x1BNReLU(in_channels=1024, out_channels=512),
            Conv3x3BNReLU(in_channels=512, out_channels= 1024),
            Conv1x1BNReLU(in_channels=1024, out_channels=512),
            Conv3x3BNReLU(in_channels=512, out_channels=1024),
            Conv3x3BNReLU(in_channels=1024, out_channels=1024),
            Conv3x3BNReLU(in_channels=1024, out_channels=1024, stride=2),
            Conv3x3BNReLU(in_channels=1024, out_channels=1024),
            Conv3x3BNReLU(in_channels=1024, out_channels=1024),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1470),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


if __name__ == '__main__':
    model = YOLO()
    print(model)

    data = torch.randn(1,3,448,448)
    output = model(data)
    print(output.shape)
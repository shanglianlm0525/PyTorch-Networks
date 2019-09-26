import torch
import torch.nn as nn
import torchvision

class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        resnet = torchvision.models.resnet50()
        layers = list(resnet.children())

        self.layer1 = nn.Sequential(*layers[:5])
        self.layer2 = nn.Sequential(*layers[5])
        self.layer3 = nn.Sequential(*layers[6])
        self.layer4 = nn.Sequential(*layers[7])

        self.lateral5 = nn.Conv2d(in_channels=2048,out_channels=256,kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.lateral2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)

        self.upsample2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample4 = nn.ConvTranspose2d(in_channels=256,out_channels=256, kernel_size=4, stride=2, padding=1)

        self.smooth2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c2 = x = self.layer1(x)
        c3 = x = self.layer2(x)
        c4 = x = self.layer3(x)
        c5 = x = self.layer4(x)

        p5 = self.lateral5(c5)
        p4 = self.upsample4(p5)+ self.lateral4(c4)
        p3 = self.upsample3(p4)+ self.lateral3(c3)
        p2 = self.upsample2(p3)+ self.lateral2(c2)

        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth4(p2)
        return p2,p3,p4,p5

if __name__ == '__main__':
    model = FPN()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    p2, p3, p4, p5 = model(input)
    print(p2.shape)
    print(p3.shape)
    print(p4.shape)
    print(p5.shape)
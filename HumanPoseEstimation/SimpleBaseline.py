import torch
import torch.nn as nn
import torchvision


class ResBlock(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class SimpleBaseline(nn.Module):
    def __init__(self, nJoints):
        super(SimpleBaseline, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResBlock, 64, 3)
        self.layer2 = self._make_layer(ResBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(ResBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(ResBlock, 512, 3, stride=2)

        self.deconv_layers = self._make_deconv_layer()
        self.final_layer = nn.Conv2d(in_channels=256,out_channels=nJoints,kernel_size=1,stride=1,padding=0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


    def _make_deconv_layer(self):
        layers = []
        for i in range(3):
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes,out_channels=256,kernel_size=4,
                                             stride=2,padding=1,output_padding=0,bias=False))
            layers.append(nn.BatchNorm2d(256))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = 256
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

if __name__ == '__main__':
    model = SimpleBaseline(nJoints=16)
    print(model)

    data = torch.randn(1,3,256,192)
    out = model(data)
    print(out.shape)

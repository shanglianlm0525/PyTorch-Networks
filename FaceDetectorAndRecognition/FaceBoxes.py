import torch
import torch.nn as nn
import torchvision

class Conv2dCReLU(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(Conv2dCReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.bn(self.conv(x))
        out = torch.cat([x, -x], dim=1)
        return self.relu(out)


class InceptionModules(nn.Module):
    def __init__(self):
        super(InceptionModules, self).__init__()

        self.branch1_conv1 = nn.Conv2d(in_channels=128,out_channels=32,kernel_size=1,stride=1)
        self.branch1_conv1_bn = nn.BatchNorm2d(32)

        self.branch2_pool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.branch2_conv1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1)
        self.branch2_conv1_bn = nn.BatchNorm2d(32)

        self.branch3_conv1 = nn.Conv2d(in_channels=128, out_channels=24, kernel_size=1, stride=1)
        self.branch3_conv1_bn = nn.BatchNorm2d(24)
        self.branch3_conv2 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch3_conv2_bn = nn.BatchNorm2d(32)

        self.branch4_conv1 = nn.Conv2d(in_channels=128, out_channels=24, kernel_size=1, stride=1)
        self.branch4_conv1_bn = nn.BatchNorm2d(24)
        self.branch4_conv2 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch4_conv2_bn = nn.BatchNorm2d(32)
        self.branch4_conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch4_conv3_bn = nn.BatchNorm2d(32)


    def forward(self, x):
        x1 = self.branch1_conv1_bn(self.branch1_conv1(x))
        x2 = self.branch2_conv1_bn(self.branch2_conv1(self.branch2_pool(x)))
        x3 = self.branch3_conv2_bn(self.branch3_conv2(self.branch3_conv1_bn(self.branch3_conv1(x))))
        x4 = self.branch4_conv3_bn(self.branch4_conv3(self.branch4_conv2_bn(self.branch4_conv2(self.branch4_conv1_bn(self.branch4_conv1(x))))))
        out = torch.cat([x1, x2, x3, x4],dim=1)
        return out

class FaceBoxes(nn.Module):
    def __init__(self, num_classes, phase):
        super(FaceBoxes, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        self.RapidlyDigestedConvolutionalLayers = nn.Sequential(
            Conv2dCReLU(in_channels=3,out_channels=24,kernel_size=7,stride=4,padding=3),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            Conv2dCReLU(in_channels=48,out_channels=64,kernel_size=5,stride=2,padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        )

        self.MultipleScaleConvolutionalLayers = nn.Sequential(
            InceptionModules(),
            InceptionModules(),
            InceptionModules(),
        )

        self.conv3_1 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=1,stride=1)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)

        self.loc_layer1 = nn.Conv2d(in_channels=128, out_channels=21*4, kernel_size=3, stride=1, padding=1)
        self.conf_layer1 = nn.Conv2d(in_channels=128, out_channels=21*num_classes, kernel_size=3, stride=1, padding=1)

        self.loc_layer2 = nn.Conv2d(in_channels=256, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conf_layer2 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=3, stride=1, padding=1)

        self.loc_layer3 = nn.Conv2d(in_channels=256, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conf_layer3 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=3, stride=1, padding=1)

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
        elif self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.xavier_normal_(m.weight.data)
                        nn.init.constant_(m.bias, 0)
                    else:
                        nn.init.xavier_normal_(m.weight.data)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.RapidlyDigestedConvolutionalLayers(x)
        out1 = self.MultipleScaleConvolutionalLayers(x)
        out2 = self.conv3_2(self.conv3_1(out1))
        out3 = self.conv4_2(self.conv4_1(out2))

        loc1 = self.loc_layer1(out1)
        conf1 = self.conf_layer1(out1)

        loc2 = self.loc_layer2(out2)
        conf2 = self.conf_layer2(out2)

        loc3 = self.loc_layer3(out3)
        conf3 = self.conf_layer3(out3)

        locs = torch.cat([loc1.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1),
                          loc2.permute(0, 2, 3, 1).contiguous().view(loc2.size(0), -1),
                          loc3.permute(0, 2, 3, 1).contiguous().view(loc3.size(0), -1)], dim=1)
        confs = torch.cat([conf1.permute(0, 2, 3, 1).contiguous().view(conf1.size(0), -1),
                           conf2.permute(0, 2, 3, 1).contiguous().view(conf2.size(0), -1),
                           conf3.permute(0, 2, 3, 1).contiguous().view(conf3.size(0), -1)], dim=1)

        if self.phase == 'test':
            out = (locs.view(locs.size(0), -1, 4),
                   self.softmax(confs.view(-1, self.num_classes)))
        else:
            out = (locs.view(locs.size(0), -1, 4),
                   confs.view(-1, self.num_classes))
        return out


if __name__ == '__main__':
    model = FaceBoxes(num_classes=2, phase='train')
    print(model)

    input = torch.randn(1, 3, 1024, 1024)
    out = model(input)
    print(out[0].shape)
    print(out[1].shape)


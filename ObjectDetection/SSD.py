import torch
import torch.nn as nn
import torchvision
import cv2

def Conv3x3BNReLU(in_channels,out_channels,stride,padding=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def Conv3x3ReLU(in_channels,out_channels,stride,padding=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU6(inplace=True)
        )

def ConvTransBNReLU(in_channels,out_channels,kernel_size,stride):
    return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def Conv1x1BNReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

class SSD(nn.Module):
    def __init__(self, phase='train', num_classes=21):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        self.detector1 = nn.Sequential(
            Conv3x3BNReLU(in_channels=3, out_channels=64, stride=1),
            Conv3x3BNReLU(in_channels=64, out_channels=64, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Conv3x3BNReLU(in_channels=64, out_channels=128, stride=1),
            Conv3x3BNReLU(in_channels=128, out_channels=128, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Conv3x3BNReLU(in_channels=128, out_channels=256, stride=1),
            Conv3x3BNReLU(in_channels=256, out_channels=256, stride=1),
            Conv3x3BNReLU(in_channels=256, out_channels=256, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Conv3x3BNReLU(in_channels=256, out_channels=512, stride=1),
            Conv3x3BNReLU(in_channels=512, out_channels=512, stride=1),
            Conv3x3BNReLU(in_channels=512, out_channels=512, stride=1),
        )

        self.detector2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Conv3x3BNReLU(in_channels=512, out_channels=512, stride=1),
            Conv3x3BNReLU(in_channels=512, out_channels=512, stride=1),
            Conv3x3BNReLU(in_channels=512, out_channels=512, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            ConvTransBNReLU(in_channels=512, out_channels=1024, kernel_size=3, stride=2),
            Conv1x1BNReLU(in_channels=1024, out_channels=1024),
            )

        self.detector3 = nn.Sequential(
            Conv1x1BNReLU(in_channels=1024, out_channels=256),
            Conv3x3BNReLU(in_channels=256, out_channels=512, stride=2),
        )

        self.detector4 = nn.Sequential(
            Conv1x1BNReLU(in_channels=512, out_channels=128),
            Conv3x3BNReLU(in_channels=128, out_channels=256, stride=2),
        )

        self.detector5 = nn.Sequential(
            Conv1x1BNReLU(in_channels=256, out_channels=128),
            Conv3x3ReLU(in_channels=128, out_channels=256, stride=1, padding=0),
        )

        self.detector6 = nn.Sequential(
            Conv1x1BNReLU(in_channels=256, out_channels=128),
            Conv3x3ReLU(in_channels=128, out_channels=256, stride=1, padding=0),
        )

        self.loc_layer1 = nn.Conv2d(in_channels=512, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer1 = nn.Conv2d(in_channels=512, out_channels=4 * num_classes, kernel_size=3, stride=1, padding=1)

        self.loc_layer2 = nn.Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer2 = nn.Conv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, stride=1, padding=1)

        self.loc_layer3 = nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer3 = nn.Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, stride=1, padding=1)

        self.loc_layer4 = nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer4 = nn.Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, stride=1, padding=1)

        self.loc_layer5 = nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer5 = nn.Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, stride=1, padding=1)

        self.loc_layer6 = nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer6 = nn.Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, stride=1, padding=1)

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
        feature_map1 = self.detector1(x)
        feature_map2 = self.detector2(feature_map1)
        feature_map3 = self.detector3(feature_map2)
        feature_map4 = self.detector4(feature_map3)
        feature_map5 = self.detector5(feature_map4)
        out = feature_map6 = self.detector6(feature_map5)

        loc1 = self.loc_layer1(feature_map1)
        conf1 = self.conf_layer1(feature_map1)

        loc2 = self.loc_layer2(feature_map2)
        conf2 = self.conf_layer2(feature_map2)

        loc3 = self.loc_layer3(feature_map3)
        conf3 = self.conf_layer3(feature_map3)

        loc4 = self.loc_layer4(feature_map4)
        conf4 = self.conf_layer4(feature_map4)

        loc5 = self.loc_layer5(feature_map5)
        conf5 = self.conf_layer5(feature_map5)

        loc6 = self.loc_layer6(feature_map6)
        conf6 = self.conf_layer6(feature_map6)

        locs = torch.cat([loc1.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1),
                          loc2.permute(0, 2, 3, 1).contiguous().view(loc2.size(0), -1),
                          loc3.permute(0, 2, 3, 1).contiguous().view(loc3.size(0), -1),
                          loc4.permute(0, 2, 3, 1).contiguous().view(loc4.size(0), -1),
                          loc5.permute(0, 2, 3, 1).contiguous().view(loc5.size(0), -1),
                          loc6.permute(0, 2, 3, 1).contiguous().view(loc6.size(0), -1)], dim=1)
        confs = torch.cat([conf1.permute(0, 2, 3, 1).contiguous().view(conf1.size(0), -1),
                           conf2.permute(0, 2, 3, 1).contiguous().view(conf2.size(0), -1),
                           conf3.permute(0, 2, 3, 1).contiguous().view(conf3.size(0), -1),
                           conf4.permute(0, 2, 3, 1).contiguous().view(conf4.size(0), -1),
                           conf5.permute(0, 2, 3, 1).contiguous().view(conf5.size(0), -1),
                           conf6.permute(0, 2, 3, 1).contiguous().view(conf6.size(0), -1)], dim=1)

        if self.phase == 'test':
            out = (locs.view(locs.size(0), -1, 4),
                   self.softmax(confs.view(confs.size(0), -1,self.num_classes)))
        else:
            out = (locs.view(locs.size(0), -1, 4),
                   confs.view(confs.size(0), -1, self.num_classes))
        return out


if __name__ == '__main__':
    model = SSD()
    print(model)

    input = torch.randn(1,3,300,300)
    output = model(input)

    print(output[0].shape)
    print(output[1].shape)

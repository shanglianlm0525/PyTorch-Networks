import torch
import torch.nn as nn
import torchvision

class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        vgg = torchvision.models.vgg16()

        features = list(vgg.features.children())

        self.padd = nn.ZeroPad2d([100,100,100,100])

        self.pool3 = nn.Sequential(*features[:17])
        self.pool4 = nn.Sequential(*features[17:24])
        self.pool5 = nn.Sequential(*features[24:])

        self.pool3_conv1x1 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.pool4_conv1x1 = nn.Conv2d(512, num_classes, kernel_size=1)

        self.output5 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, num_classes, kernel_size=1),
        )

        self.up_pool3_out = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8)
        self.up_pool4_out = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2)
        self.up_pool5_out = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2)

    def forward(self, x):
        _,_, w, h = x.size()

        x = self.padd(x)
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)

        output5 = self.up_pool5_out(self.output5(pool5))

        pool4_out = self.pool4_conv1x1(0.01 * pool4)
        output4 = self.up_pool4_out(pool4_out[:,:,5:(5 + output5.size()[2]) ,5:(5 + output5.size()[3])]+output5)

        pool3_out = self.pool3_conv1x1(0.0001 * pool3)
        output3 = self.up_pool3_out(pool3_out[:, :, 9:(9 + output4.size()[2]), 9:(9 + output4.size()[3])] + output4)

        out = self.up_pool3_out(output3)

        out = out[:, :, 31: (31 + h), 31: (31 + w)].contiguous()
        return out


if __name__ == '__main__':
    model = FCN8s(num_classes=20)
    print(model)

    input = torch.randn(1,3,224,224)
    output = model(input)
    print(output.shape)


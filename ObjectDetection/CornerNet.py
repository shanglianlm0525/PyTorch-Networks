import torch
import torch.nn as nn
import torchvision

class HourglassNetwork(nn.Module):
    def __init__(self):
        super(HourglassNetwork, self).__init__()

    def forward(self, x):
        return out

class PredictionModule(nn.Module):
    def __init__(self):
        super(PredictionModule, self).__init__()

    def forward(self, x):
        return out


class CornerNet(nn.Module):
    def __init__(self):
        super(CornerNet, self).__init__()

    def forward(self, x):
        return out


if __name__ == '__main__':
    model = CornerNet()
    print(model)

    data = torch.randn(1,3,511,511)
    output = model(data)
    print(output.shape)
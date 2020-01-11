import torch
import torch.nn as nn

def ConvBNReLU(in_channels,out_channels,kernel_size,stride,padding=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        mid_channels = out_channels//2

        self.bottleneck = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1),
            ConvBNReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1),
        )
        self.shortcut = ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.bottleneck(x)
        return out+self.shortcut(x)


class left_pool(torch.autograd.Function):
    def forward(self, input_):
        self.save_for_backward(input_.clone())
        output = torch.zeros_like(input_)
        batch = input_.size(0)
        width = input_.size(3)

        input_tmp = input_.select(3, width - 1)
        output.select(3, width - 1).copy_(input_tmp)

        for idx in range(1, width):
            input_tmp = input_.select(3, width - idx - 1)
            output_tmp = output.select(3, width - idx)
            cmp_tmp = torch.cat((input_tmp.view(batch, 1, -1), output_tmp.view(batch, 1, -1)), 1).max(1)[0]
            output.select(3, width - idx - 1).copy_(cmp_tmp.view_as(input_tmp))

        return output

    def backward(self, grad_output):
        input_, = self.saved_tensors
        output = torch.zeros_like(input_)

        grad_output = grad_output.clone()
        res = torch.zeros_like(grad_output)

        w = input_.size(3)
        batch = input_.size(0)

        output_tmp = res.select(3, w - 1)
        grad_output_tmp = grad_output.select(3, w - 1)
        output_tmp.copy_(grad_output_tmp)

        input_tmp = input_.select(3, w - 1)
        output.select(3, w - 1).copy_(input_tmp)

        for idx in range(1, w):
            input_tmp = input_.select(3, w - idx - 1)
            output_tmp = output.select(3, w - idx)
            cmp_tmp = torch.cat((input_tmp.view(batch, 1, -1), output_tmp.view(batch, 1, -1)), 1).max(1)[0]
            output.select(3, w - idx - 1).copy_(cmp_tmp.view_as(input_tmp))

            grad_output_tmp = grad_output.select(3, w - idx - 1)
            res_tmp = res.select(3, w - idx)
            com_tmp = comp(input_tmp, output_tmp, grad_output_tmp, res_tmp)
            res.select(3, w - idx - 1).copy_(com_tmp)
        return res

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
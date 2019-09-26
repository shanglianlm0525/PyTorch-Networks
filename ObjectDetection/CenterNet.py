import torch
import torch.nn as nn
import torchvision






if __name__ == '__main__':
    model = YOLO()
    print(model)

    data = torch.randn(1,3,448,448)
    output = model(data)
    print(output.shape)
import torch
import torch.nn as nn


class CustomResNet(nn.Module):
    def __init__(self, flatten_superpixels=False):
        super(CustomResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    def forward(self, x):
        return self.conv1(x)

conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet = CustomResNet()
input = torch.zeros((16, 9, 1, 105, 105))
print(f'output_shape: {resnet(input[0]).shape}')
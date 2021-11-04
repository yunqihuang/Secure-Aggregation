from torchvision import models
from torch import nn


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=(1, 1))
        self.resnet = models.resnet50(pretrained=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.resnet(x)
        return x

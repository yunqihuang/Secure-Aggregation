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


class MLP(nn.Module):
    def __init__(self, in_features=784, num_classes=10, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

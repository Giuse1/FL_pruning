from torch import nn
import torch.nn.functional as F

class cifar(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(cifar, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, bias=False)
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)

        x = F.relu(F.max_pool2d(x, 2))
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = F.relu(F.max_pool2d(x, 2))
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = self.fc2(x)
        return x




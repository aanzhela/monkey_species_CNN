import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(in_features=1024, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 64x64x3
        x = self.relu(self.conv1(x))
        # 64x64x16
        x = self.maxpool(x)
        # 32x32x16
        x = self.relu(self.conv2(x))
        # 32x32x32
        x = self.maxpool(x)
        # 16x16x32
        x = self.relu(self.conv3(x))
        # 16x16x64
        x = self.maxpool(x)
        # 8x8x64
        x = self.relu(self.conv4(x))
        # 8x8x64
        x = self.maxpool(x)
        # 4x4x64
        # Flattening for FC
        x = x.view(-1, 1024)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

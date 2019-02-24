import torch
import torch.nn as nn


class CAE(nn.Module):

    def __init__(self):
        super(CAE, self).__init__()
        # encoder
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3)
        self.down_conv1 = nn.Conv2d(3, 64, 3)
        self.down_bn1 = nn.BatchNorm2d(64)
        self.down_conv2 = nn.Conv2d(64, 128, 3)
        self.down_bn2 = nn.BatchNorm2d(128)
        self.linear = nn.Linear(512, 1024)

        # decoder
        self.up_conv1 = nn.ConvTranspose2d(1024, 512, 3)
        self.up_bn1 = nn.BatchNorm2d(512)
        self.up_conv2 = nn.ConvTranspose2d(512, 256, 3, stride=2)
        self.up_bn2 = nn.BatchNorm2d(256)
        self.up_conv3 = nn.ConvTranspose2d(256, 128, 3, stride=2)
        self.up_bn3 = nn.BatchNorm2d(128)
        self.up_conv4 = nn.ConvTranspose2d(128, 64, 4, stride=2)
        self.up_bn4 = nn.BatchNorm2d(64)
        self.mix_conv = nn.Conv2d(64, 3, 3, stride=1, padding=1)

    def encoder(self, x):
        x = self.down_conv1(x)
        x = self.down_bn1(x)
        x = self.pool(x)
        x = self.relu(x)

        x = self.down_conv2(x)
        x = self.down_bn2(x)
        x = self.pool(x)
        x = self.relu(x)

        x = x.view(len(x), -1)

        x = self.linear(x)
        x = torch.unsqueeze(x, -1)
        x = torch.unsqueeze(x, -1)

        return x

    def decoder(self, x):

        x = self.up_conv1(x)
        x = self.up_bn1(x)
        x = self.relu(x)

        x = self.up_conv2(x)
        x = self.up_bn2(x)
        x = self.relu(x)

        x = self.up_conv3(x)
        x = self.up_bn3(x)

        x = self.up_conv4(x)
        x = self.up_bn4(x)
        x = self.relu(x)
        x = self.mix_conv(x)

        return x

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

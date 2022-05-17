# coding=gbk

from torch import nn
import torch
import torch.nn.functional as F
import math


class ResBlock(nn.Module):

    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.stride = stride
        self.block = nn.Sequential(

            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=self.stride, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),

            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),

            nn.Conv2d(outchannel, outchannel * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(outchannel * 4),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel * 4),
            )

    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, ResBlock, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        self.layer1 = self._make_layer(ResBlock, 64, layers[0])
        self.layer2 = self._make_layer(ResBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(ResBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(ResBlock, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # Conv2d层初始化
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):  # BatchNorm2d初始化偏置为0，权重为1
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, ResBlock, channel, blocks, stride=1):
        layers = []
        layers.append(ResBlock(self.inchannel, channel, stride))
        self.inchannel = channel * 4
        for i in range(1, blocks):
            layers.append(ResBlock(self.inchannel, channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50():
    return ResNet(ResBlock, [3, 4, 6, 3])


def resnet101():
    return ResNet(ResBlock, [3, 4, 23, 3])


def resnet152():
    return ResNet(ResBlock, [3, 8, 36, 3])

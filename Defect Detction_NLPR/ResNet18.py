# coding=gbk

from torch import nn
import torch
import torch.nn.functional as F
import math


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=stride, bias=False),  # bias=False
            nn.BatchNorm2d(outchannel),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class resnet18(nn.Module):
    def __init__(self, ResBlock, ClassNum=2):  # AvgPl_size: 平均池化层卷积核大小 128*128图像输入为4、224*224图像输入为7
        super(resnet18, self).__init__()
        self.inchannel = 64  # 第一次卷积层后的输入段通道数
        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pooling = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.Conv2 = self.Get_layer(ResBlock, 64, 1, 2)
        self.Conv3 = self.Get_layer(ResBlock, 128, 2, 2)
        self.Conv4 = self.Get_layer(ResBlock, 256, 2, 2)
        self.Conv5 = self.Get_layer(ResBlock, 512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, ClassNum)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # Conv2d层初始化
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):  # BatchNorm2d初始化偏置为0，权重为1
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def Get_layer(self, ResBlock, channel, stride, blockNum):  # stride同通道数的几层里第一层的stride  blockNum:对应block的重复次数
        strides = [stride] + [1] * (blockNum - 1)
        layers = []
        for std in strides:
            layers.append(ResBlock(self.inchannel, channel, stride=std))
            self.inchannel = channel
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.pooling(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


'''
class ResNet50

'''


def ResNet_18():
    return resnet18(ResBlock)

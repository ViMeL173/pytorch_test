# coding=gbk

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import argparse
import os
from CLS_Dataset import CLSdataset
from CLS_Dataset import Normalize_CLS
from ResNet18 import ResNet_18
from ResNet50_101_152 import resnet50, resnet101, resnet152
import cv2

# 参数设置
EPOCH = 200  # 遍历数据集次数
BATCH_SIZE = 64  # 批处理尺寸(batch_size)
LR = 0.01  # 学习率

# 图像预处理大小
img_size = 224
# 图像预处理模式
img_mode = 1
# 使用网络层数18,50,101,152
layers = 18
# 是否加载预训练模型
Pre_model_flag = True
'''
:param mode: 0：不做处理
             1：裁剪拼接
             2：填充策略
             3：裁剪填充拼接
'''

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 计算数据库的均值和方差
train_mean, train_std, test_mean, test_std = Normalize_CLS(img_size, img_mode)
# 准备数据集并预处理


transform_train = transforms.Compose([
    transforms.Resize([img_size, img_size]),
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((round(train_mean, 3)), (round(train_std, 3))),  # 归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.Resize([img_size, img_size]),
    transforms.ToTensor(),
    transforms.Normalize((round(test_mean, 3)), (round(test_std, 3))),
])

trainset = CLSdataset(train=True, transform=transform_train, mode=img_mode)  # 训练数据集
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = CLSdataset(train=False, transform=transform_test, mode=img_mode)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

if Pre_model_flag:
    # 预训练模型
    if layers == 18:
        Pre_model = models.resnet18(pretrained=True)
    elif layers == 50:
        Pre_model = models.resnet50(pretrained=True)
    elif layers == 101:
        Pre_model = models.resnet101(pretrained=True)
    elif layers == 152:
        Pre_model = models.resnet152(pretrained=True)
    else:
        Pre_model = models.resnet18(pretrained=True)

    if layers == 18:
        Pre_model.fc = nn.Linear(512, 2)

    elif layers == 50 or 101 or 152:
        Pre_model.fc = nn.Linear(512 * 4, 2)

    else:
        Pre_model.fc = nn.Linear(512, 2)

    Pre_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = Pre_model

else:
    # 自己的模型
    if layers == 18:
        model = ResNet_18()
    elif layers == 50:
        model = resnet50()
    elif layers == 101:
        model = resnet101()
    elif layers == 152:
        model = resnet152()
    else:
        model = ResNet_18()

model.to(device)
criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                      weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

loss_list = []
acc_list = []


def train(epoch):
    running_loss = 0.0
    global loss_list
    if epoch == 50:
        for p in optimizer.param_groups:
            p['lr'] = 0.001
    for idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if idx % 10 == 9:  # 每30论输出一个loss
            print('[%d, %5d] loss: %.3f' % (epoch + 1, idx + 1, running_loss / 10))
            loss_list.append(running_loss / 10)
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    global acc_list
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # 取一行中最大值的下标号
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 计算正确的个数
    print('正确率: %d %%' % (100 * correct / total))
    acc_list.append(100 * correct / total)


if __name__ == '__main__':
    for epoch in range(EPOCH):
        train(epoch)
        test()
        if epoch % 16 == 15:
            print("acc_list:", acc_list)
            print("loss_list:", loss_list)
    print("acc_list:", acc_list)
    print("loss_list:", loss_list)

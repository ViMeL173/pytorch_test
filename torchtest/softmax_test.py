# coding=gbk

import torch
import cv2
from CNN_test import Net
from torchvision import transforms


def identity_num(img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('model.pth')  # 加载模型
    model = model.to(device)
    model.eval()  # 把模型转为test模式
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    output = model(img)
    _, predicted = torch.max(output.data, dim=1)  # 取一行中最大值的下标号
    print(predicted.item())


if __name__ == '__main__':
    img = cv2.imread("mytest.png")  # 读取要预测的图片
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片转为灰度图，因为mnist数据集都是灰度图
    identity_num(img)

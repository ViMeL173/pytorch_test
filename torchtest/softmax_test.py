# coding=gbk

import torch
import cv2
from CNN_test import Net
from torchvision import transforms


def identity_num(img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('model.pth')  # ����ģ��
    model = model.to(device)
    model.eval()  # ��ģ��תΪtestģʽ
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    output = model(img)
    _, predicted = torch.max(output.data, dim=1)  # ȡһ�������ֵ���±��
    print(predicted.item())


if __name__ == '__main__':
    img = cv2.imread("mytest.png")  # ��ȡҪԤ���ͼƬ
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # ͼƬתΪ�Ҷ�ͼ����Ϊmnist���ݼ����ǻҶ�ͼ
    identity_num(img)

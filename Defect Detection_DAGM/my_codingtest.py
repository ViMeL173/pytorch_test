# coding=gbk


import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from Module import Net
from DAGM2007 import DAGM2007
import matplotlib.pyplot as plt

# input = [3, 4, 6, 5, 7,
#          2, 4, 6, 8, 2,
#          1, 6, 7, 8, 4,
#          9, 7, 4, 6, 2,
#          3, 7, 5, 4, 1]
# input = torch.Tensor(input).view(1, 1, 5, 5)
# print("input:", input)
# conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)#, bias=False)
# kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)
# print("kenel:", kernel)
# conv_layer.weight.data = kernel.data
# output = conv_layer(input)
# print("output:", output)

# img = cv2.imread(r'D:\python_pro\Datasets\DAGM\Class1\Train\0576.PNG', 0)
# img2 = cv2.imread(r'D:\python_pro\Datasets\DAGM\Class1\Train\0577.PNG', 0)
# img3 = cv2.imread(r'D:\python_pro\Datasets\DAGM\Class1\Train\0578.PNG', 0)
# bacth = img.size(0)
# img = torch.Tensor([img, img2, img3])
# print(img)
#
# print(bacth)
# img = img.view(bacth, -1)
# print(img)
# debug = 1
# if debug == 1:
#     labels = list()
#     with open("D:\python_pro\Datasets\DAGM\Class1\Train\Label\Labels.txt", "r") as f:
#         lines = f.readlines()
#     print(lines)
#     for line in lines:
#         if line == '1\n':
#             continue
#         # all_label = line.split()
#         # label = cv2.imread(r"D:\python_pro\Datasets\DAGM\Class1\Train\{}".format(all_label[2]), 0)
#         # labels.append(label)
#         labels.append(line.strip() + '9')
#     print(labels)
#     a = labels[0]
#     path = a.split()[2]
#     l = a.split()[4]
#     print("img,label", path, l)
#

# from torch.utils.data import Dataset
#
#
# class DAGM2007(Dataset):
#     def __init__(self, dataset_type='Train', transform=None, update_dataset=False):
#         """
#         dataset_type: Train or Test (必须大写)
#         """
#
#         dataset_path = 'D:/python_pro/python_proj/pytorch_test/dataset/DAGM/'
#
#         if update_dataset:
#             pass
#         self.transform = transform
#         self.sample_list = list()
#         self.dataset_type = dataset_type
#         for i in range(1, 7):
#             with open(dataset_path + 'Class{}/'.format(i) + self.dataset_type + '/Label/labels.txt') as f:  # 锁定目标文件
#                 lines = f.readlines()
#                 img_path = dataset_path + 'Class{}/'.format(i) + self.dataset_type + '/' + '\t'  # 图像文件前置路径
#                 for line in lines:
#                     if line == '1\n':
#                         continue  # 去除第一行的“1”
#                     self.sample_list.append(img_path + line.strip() + '\t{}'.format(i))  # 读取txt内容并保存于list中(i为class标签)
#
#     def __getitem__(self, index):
#         item = self.sample_list[index]
#         img = Image.open(item.split()[0] + item.split()[3])
#         if self.transform is not None:
#             img = self.transform(img)
#         label = int(item.split()[6])
#         return img, label
#
#     def __len__(self):
#         return len(self.sample_list)
#
#     def get_name(self, index):
#         item = self.sample_list[index]
#         return item.split()[3]
#
#
# if __name__ == '__main__':
#     batch_size = 3450
#     transform = transforms.Compose(
#         [transforms.Resize([128, 128]), transforms.ToTensor()])
#     dagm_train = DAGM2007(transform=transform, train=True)
#     train_loader = DataLoader(dagm_train, shuffle=True, batch_size=batch_size)
#     for idx, data in enumerate(train_loader, 0):
#         ipt, tg = data
#         x = ipt.view(-1, 128 * 128)
#         x_std = x.std().item()
#         x_mean = x.mean().item()
#     print('均值：', x_mean)
#     print('标准差：', x_std)

# dagm = DAGM2007(transform=transform)
# img, gt = dagm.__getitem__(19)
# name = dagm.get_name(19)
# print(gt)
# print(name)

# x = torch.randn(1, 1024)
# fc = torch.nn.Linear(1024, 512)
# print(fc.weight.size())
# w = torch.randn(512, 1024)
# w1 = torch.nn.init.trunc_normal_(w, mean=0, std=1)
# fc.weight = Parameter(w1)
# x = fc(x)
# print(x.size())

# def Learning_rate_attenuation(lr, idx):
#     return lr / (1 + 0.012 * idx)
#
#
# model = Net()
# LR = 0.1
# optimizer = torch.optim.SGD(model.parameters(), weight_decay=5e-5, lr=LR, momentum=0.9)
# lr_list = []
# for epoch in range(100):
#     for p in optimizer.param_groups:
#         p['lr'] = Learning_rate_attenuation(p['lr'], epoch)
#     lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
#
# plt.plot(range(100), lr_list, color='r')
# plt.show()
# print(lr_list)

#
# batch_size = 50
# transform = transforms.Compose(
#     [transforms.Resize([128, 128]), transforms.ToTensor()])
# dagm_train = DAGM2007(transform=transform, train=True)
# train_loader = DataLoader(dagm_train, shuffle=True, batch_size=batch_size)
# dagm_test = DAGM2007(transform=transform, train=False)
# test_loader = DataLoader(dagm_test, shuffle=False, batch_size=batch_size)
# for idx, data in enumerate(train_loader, 0):
#     inputs, target = data
#     print(target)

# acc_list = [32.84057971014493, 90.08695652173913, 98.98550724637681, 99.76811594202898, 99.71014492753623,
#             97.42028985507247, 99.65217391304348, 99.68115942028986, 99.76811594202898, 99.85507246376811]
# loss_list = [1.7856299797693889, 1.7633850932121278, 1.5718759218851726, 0.7941080033779144, 0.18718916575113934,
#              0.07839145213365555, 0.025347198406234384, 0.015636218370248874, 0.01710024242347572, 0.010285383412459244,
#              0.027078440009305874, 0.08930976374540478, 0.03551232501049526, 0.022661255125422032, 0.01861055419431068,
#              0.008546723570907489, 0.008739728106108183, 0.007502457239509871, 0.001526923303026706,
#              0.0009341683631646447]
#
# plt.plot(range(len(loss_list)), loss_list, color='r')
# plt.show()

# def cv_show(img):
#     cv2.imshow('image', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# img = cv2.imread("D:/python_pro/Datasets/cls_data/images/cam-0_20220221_142739.jpg", 0)
# print(img.shape)
# cv_show(img)
# img = cv2.resize(img, (138, 138))
# cv_show(img)
# sample_list = list()
# with open("D:/python_pro/Datasets/cls_data/train.txt") as f:  # 锁定目标文件
#     lines = f.readlines()
#     img_path = "D:/python_pro/Datasets/cls_data/" + 'images' + '/'  # 图像文件前置路径
#     for line in lines:
#         sample_list.append(
#             img_path + line.strip())  # 读取txt内容并保存于list中(i为class标签)
#
# # list = sample_list.split()
# print(sample_list[0].split())
# img = Image.open(sample_list[0].split()[0])
# label = sample_list[0].split()[1]
# print(label)


# coding=gbk



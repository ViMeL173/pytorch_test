# coding=gbk
# coding:utf-8
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import *
import os


def rotation(img, mode):  # 图像方位设定
    '''
    :param img:  输入图片
    :param mode: 1 竖版  0 横板
    :return: 输出
    '''
    height, width = img.shape
    if mode == 1:
        if height <= width:
            img2 = np.zeros([width, height])
            for x in range(height):
                for y in range(width):
                    img2[y, x] = img[x, y]
            return img2
        else:
            return img
    else:
        if height > width:
            img2 = np.zeros([width, height])
            for x in range(height):
                for y in range(width):
                    img2[y, x] = img[x, y]
            return img2
        else:
            return img


def imshow(img):
    plt.suptitle("img")
    plt.imshow(img, "gray")
    plt.show()


path = "D:/python_pro/Datasets/anno_v220420/"
target_path = "D:/python_pro/python_proj/yolov5-master/data/defect_data/"

flag = "test"
n = 1
with open(path + "val.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        imgname = line.split()[0]
        img = cv2.imread(path + "roi_images/" + imgname, 0)
        cv2.imwrite(target_path + "images/" + flag + "/" + imgname, img)
        txtname = imgname.replace(".jpg", ".txt")
        os.chdir(target_path + "labels/" + flag)
        shutil.copy(path + "yolo_labels/" + txtname, target_path + "labels/" + flag)
        print("copy successful:", n, imgname)
        n += 1

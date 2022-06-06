# coding=utf-8

import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm
import shutil
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt

seq_bri = iaa.Sequential([
    iaa.MultiplyBrightness((1.4, 1.8)),  # 亮度增强
])

seq_Motionblur = iaa.Sequential([
    iaa.MotionBlur(k=10)
])


def pltshow(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def getbright(p, thr):
    flag = 1
    light = round(p[0] * 0.299 + p[1] * 0.587 + p[2] * 0.114, 3)
    if light > thr:
        flag = 0
    return flag


def brightness(img, bri=40, thr=200):
    (row, col, chs) = img.shape
    for j in range(row):
        for i in range(col):
            flag = 0
            if getbright(img[j][i][:], thr):
                if img[j][i][0] + bri > 255:
                    img[j][i][0] = 255
                    flag = 1
                if img[j][i][1] + bri > 255:
                    img[j][i][1] = 255
                    flag = 1
                if img[j][i][2] + bri > 255:
                    img[j][i][2] = 255
                    flag = 1
                if flag:
                    continue
                else:
                    img[j][i][:] += bri
    return img


model = 5

path = "/home/wushangfeng/module/mmsegmentation/data/acdc_challenge/images/validation/"
path2 = "/home/wushangfeng/module/mmsegmentation/data/acdc_challenge/images/validation_imgaug/"
# allpaths = os.listdir(path)
# img_paths = []
# pathlists = [os.path.join(path, allpath) for allpath in allpaths]
# for pathlist in pathlists:
#     imglists = glob(pathlist + '/*.png')
#     for imglist in imglists:
#         img_paths.append(imglist)
img_paths = glob(path + '*.png')

for img_path in tqdm(img_paths, desc="changing", mininterval=0.1):
    img = cv2.imread(img_path)
    imgname = os.path.basename(img_path)
    target_path = os.path.join(path2, imgname)
    if model == 1:
        if 'GOPR0351' in img_path:
            img = brightness(img, 50, 255)
            cv2.imwrite(target_path, img)
        elif 'GOPR0356' in img_path:
            img = brightness(img, 50, 255)
            cv2.imwrite(target_path, img)
        else:
            shutil.copyfile(img_path, target_path)
    elif model == 2:
        if 'GOPR0351' in img_path:
            img_aug = seq_bri.augment_image(img)
            img_aug = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)
            cv2.imwrite(target_path, img_aug)
        elif 'GOPR0356' in img_path:
            img_aug = seq_bri.augment_image(img)
            img_aug = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)
            cv2.imwrite(target_path, img_aug)
        else:
            shutil.copyfile(img_path, target_path)
    elif model == 3:
        img_aug = seq_Motionblur.augment_image(img)
        cv2.imwrite(target_path, img_aug)
    elif model == 4:  # night
        if 'GOPR0351' in img_path:
            continue
        elif 'GOPR0356' in img_path:
            continue
        else:
            shutil.copyfile(img_path, target_path)
    elif model == 5:  # fog
        if 'GOPR0476' in img_path:
            continue
        elif 'GP010476' in img_path:
            continue
        elif 'GP020475' in img_path:
            continue
        else:
            shutil.copyfile(img_path, target_path)
# if 'GOPR0356' in img_path:
#     img = brightness(img, 40, 255)
#     cv2.imwrite(target_path, img)
# elif 'GOPR0355' in img_path:
#     img = brightness(img, 40, 255)
#     cv2.imwrite(target_path, img)
# elif 'GOPR0364' in img_path:
#     img = brightness(img, 40, 255)
#     cv2.imwrite(target_path, img)
# elif 'GOPR0594' in img_path:
#     img = brightness(img, 40, 255)
#     cv2.imwrite(target_path, img)
# elif 'GP010364' in img_path:
#     img = brightness(img, 40, 255)
#     cv2.imwrite(target_path, img)
# elif 'GP010594' in img_path:
#     img = brightness(img, 40, 255)
#     cv2.imwrite(target_path, img)

# os.system(
#     'python tools/test.py /data/share/ACDC_Models/model_v220601/220527_2_L_area_resize.py /data/share/ACDC_Models/model_v220601/iter_80000.pth --work-dir checkpoint --eval mIoU')

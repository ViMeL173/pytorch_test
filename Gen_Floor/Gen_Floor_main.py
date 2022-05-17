# coding=gbk

import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import *
import glob

label_map = {'bg': 0, 'floor': 1, 'light': 2, 'defect': 3}
image_flod = "D:/python_pro/Datasets/anno_data_v220415/half_images/"   # 数据集图片路径
texture_path = "D:/python_pro/Datasets/anno_data_v220415/Texture/"     # 纹理路径


def cv_show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def imshow(img):
    plt.suptitle("img")
    plt.imshow(img, "gray")
    plt.show()


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


def Gen_tex_mask(tex, mask, res=1):  # 随即裁剪一块非背景区域大小的纹理图片
    # 索引第一位为y，然后是x ——> img[y,x]
    """
    :param tex:  纹理图
    :param mask:  label mask
    :param res:  纹理图放大倍数，默认不放大
    :return:
    """
    height, width = tex.shape
    tex = rotation(tex, 1)
    region_locs = np.where(mask != label_map['bg'])
    xlen = max(region_locs[1]) - min(region_locs[1]) + 1
    ylen = max(region_locs[0]) - min(region_locs[0]) + 1
    if xlen > min(height, width):
        tex = cv2.resize(tex, (int(width * 2), int(height * 2)))
    height2, width2 = tex.shape
    tex = cv2.resize(tex, (int(width2 * res), int(height2 * res)))
    tex_mask = np.zeros_like(mask)
    x_start = randint(0, (tex.shape[1] - xlen))
    y_start = randint(0, (tex.shape[0] - ylen))
    rand_crop = tex[y_start:y_start + ylen, x_start:x_start + xlen]
    tex_mask[min(region_locs[0]):max(region_locs[0]) + 1, min(region_locs[1]):max(region_locs[1]) + 1] = rand_crop
    tex_mask[mask == 0] = 255
    return tex_mask


def Get_R(tex_mean):  # 获取导向滤波半径
    if tex_mean < 80:
        R = 20
    elif 80 <= tex_mean < 100:
        R = 16
    elif 100 <= tex_mean < 120:
        R = 12
    else:
        R = 10
    return R


image_paths = glob.glob(image_flod + '*.jpg')

if __name__ == '__main__':
    for image_path in image_paths:
        mask_path = image_path.replace('half_images', 'half_segms').replace('.jpg', '.png')   # label mask路径
        image = cv2.imread(image_path, 0)
        mask = cv2.imread(mask_path, 0)
        tex_num = randint(1, 18)
        texture_image = cv2.imread(texture_path + "wenli{}.png".format(tex_num), 0)   # 随机选择纹理

        if (mask != label_map["bg"]).sum() != 0:
            tex_mean = np.mean(texture_image)
            R = Get_R(tex_mean)
            print("image_path:", image_path)
            print("mean:", tex_mean)
            print("R:", R)
            ori_img = image.copy()
            height, width = image.shape
            texture_image = Gen_tex_mask(texture_image, mask)
            image[mask == 0] = 255

            texture_image[mask == 0] = 255
            img_mix_w = cv2.addWeighted(image, 0.8, texture_image, 0.2, gamma=5)
            img_mix2 = cv2.ximgproc.guidedFilter(guide=image, src=img_mix_w, radius=R, eps=0.01, dDepth=-2)
            img_mix2[mask == 0] = ori_img[mask == 0]

            # random floor
            floor_map_ratio = np.random.uniform() * 0.2 + 0.7
            img_mix2 = img_mix2.astype(np.float32)
            img_mix2[mask != 0] *= floor_map_ratio
            img_mix2[img_mix2 > 255] = 255
            img_mix2[img_mix2 < 0] = 0
            img_mix2 = img_mix2.astype(np.uint8)
            imshow(img_mix2)

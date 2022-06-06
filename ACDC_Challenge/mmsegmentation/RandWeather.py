# coding=utf-8

from imgaug import augmenters as iaa
import os, cv2
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import math
import random


def pltshow(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def cv_show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def GenWeather(img, mask):
    seq_fog2 = iaa.Sequential([
        iaa.CloudLayer(
            intensity_mean=(240, 255),
            intensity_freq_exponent=(-2.0, -1.5),
            intensity_coarse_scale=10,
            alpha_min=(0.7, 0.9),
            alpha_multiplier=0.3,
            alpha_size_px_max=(2, 4),
            alpha_freq_exponent=(-4.0, -2.0),
            sparsity=0.9,
            density_multiplier=(0.5, 0.7),
            seed=None, name=None,
            random_state='deprecated', deterministic='deprecated'
        )
    ])
    # intensity_mean = (196, 255),
    # intensity_freq_exponent = (-2.5, -2.0),
    # intensity_coarse_scale = 10,
    # alpha_min = 0,
    # alpha_multiplier = (0.25, 0.75),
    # alpha_size_px_max = (2, 8),
    # alpha_freq_exponent = (-2.5, -2.0),
    # sparsity = (0.8, 1.0),
    # density_multiplier = (0.5, 1.0),
    # seed = seed,
    # random_state = random_state,
    # deterministic = deterministic
    seq_fog = iaa.Sequential([
        iaa.MultiplyBrightness((0.7, 0.8)),  # 亮度降低
        iaa.GammaContrast((0.5, 0.7)),  # 对比度降低
        iaa.ChangeColorTemperature((5300, 5900))  # 冷色调
        # iaa.Fog()
    ])
    seq_skyfog = iaa.Sequential([
        iaa.RemoveSaturation(1.0),
    ])

    seq_rain = iaa.Sequential([
        iaa.Rain(),
    ])

    seq_snow = iaa.Sequential([
        iaa.FastSnowyLandscape(
            lightness_threshold=[10, 180],
            lightness_multiplier=(2.0, 4.0))
    ])
    seq_snow_2 = iaa.Sequential([
        iaa.Snowflakes()
    ])
    seq_snowbg = iaa.Sequential([
        iaa.MultiplyBrightness((0.7, 0.8)),  # 亮度降低
        iaa.MultiplyHueAndSaturation(mul_saturation=(0.6, 0.8)),  # 饱和度降低
        iaa.GammaContrast((0.4, 0.6)),  # 对比度降低
        iaa.ChangeColorTemperature((5200, 6000))  # 冷色调
        # iaa.Fog()
    ])
    flag = 3  # np.random.randint(3)
    if flag == 0:  # rain
        mask_sky = np.zeros_like(mask)
        mask_sky[mask == 10] = 255
        k = np.ones((20, 20), np.uint8)
        mask_sky = cv2.dilate(mask_sky, k)
        img_aug = seq_skyfog.augment_image(img)
        img[mask_sky == 255] = img_aug[mask_sky == 255]
        img_aug = seq_rain.augment_image(img)

    elif flag == 1:  # fog
        mask_road = np.zeros_like(mask)
        mask_road[mask == 0] = 255
        kr = np.ones((50, 50), np.uint8)
        mask_road = cv2.dilate(mask_road, kr)

        mask_sky = np.zeros_like(mask)
        mask_sky[mask == 10] = 255
        k = np.ones((20, 20), np.uint8)
        mask_sky = cv2.dilate(mask_sky, k)

        (row, col, chs) = img.shape
        if 255 in mask_road:  # 以道路为基准设置雾化中心
            r_c = 0
            for i in range(row):
                mask1 = mask_road[0:i, 0:col]
                if 255 in mask1:
                    r_c += 1
                    if r_c >= 10:
                        row_center = i
                        break
            center_list = mask_road[row_center, 0:col]
            col_list = []
            for j in range(col):
                if center_list[j] == 255:
                    col_list.append(j)
            col_center = sum(col_list) // len(col_list)
        else:
            row_center = row // 4
            col_center = col // 2

        img_f = img / 255.0
        A = 0.2  # 亮度
        beta = random.uniform(0.03, 0.04)  # 雾的浓度
        size = math.sqrt(max(row, abs(col - col_center) * 2, col, col_center * 2)) + random.randint(0, 10)  # 雾化尺寸
        center = (row_center, col_center)  # 雾化中心
        for j in range(row):
            for l in range(col):
                d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
                td = math.exp(-beta * d)
                img_f[j][l][:] = (img_f[j][l][:] * td + A * (1 - td)) * 255
        img = img_f.astype(np.uint8)

        img_cpy = img.copy()
        img_aug = seq_skyfog.augment_image(img_cpy)
        img_aug[mask_sky == 0] = img[mask_sky == 0]
        img_aug = seq_fog.augment_image(img_aug)

    elif flag == 2:  # snow
        mask_sky = np.zeros_like(mask)
        mask_sky[mask == 10] = 255
        k = np.ones((20, 20), np.uint8)
        mask_sky = cv2.dilate(mask_sky, k)

        img_aug = seq_skyfog.augment_image(img)
        img[mask_sky == 255] = img_aug[mask_sky == 255]

        mask_cpy = np.zeros_like(mask)
        mask_cpy[mask == 0] = 255
        mask_cpy[mask == 1] = 255
        mask_cpy[mask == 9] = 255
        kr = np.ones((10, 10), np.uint8)
        mask_cpy = cv2.dilate(mask_cpy, kr)
        img_cpy = img.copy()
        img_aug = seq_snow.augment_image(img_cpy)
        img_aug[mask_cpy == 0] = img[mask_cpy == 0]
        img_aug = seq_snow_2.augment_image(img_aug)
        img_aug = seq_snowbg.augment_image(img_aug)
    else:
        mask1 = mask.copy()
        mask = np.zeros_like(img)
        mask[:, :, 0] = mask1
        mask[:, :, 1] = mask1
        mask[:, :, 2] = mask1
        mask_sky = np.zeros_like(mask)
        mask_sky[mask == 10] = 255
        k = np.ones((20, 20), np.uint8)
        mask_sky = cv2.dilate(mask_sky, k)

        img_aug = seq_skyfog.augment_image(img)
        img[mask_sky == 255] = img_aug[mask_sky == 255]

        mask_notroad = np.zeros_like(mask)
        mask_notroad[mask == 10] = 255
        knr = np.ones((50, 100), np.uint8)
        mask_notroad = cv2.dilate(mask_notroad, knr)
        mask_notroad2 = mask_notroad
        for i in range(50):
            mask_notroad2 = cv2.GaussianBlur(mask_notroad2, (49, 17), 10)
        for i in range(50):
            mask_notroad2 = cv2.GaussianBlur(mask_notroad2, (17, 49), 10)
        mask_notroad = 255 - mask_notroad2
        img_fog = cv2.addWeighted(img.copy(), 0.3, mask_notroad, 0.7, 1)
        img_aug = seq_fog2.augment_image(img_fog)
        img_aug = cv2.addWeighted(img_aug, 0.5, mask_notroad2, 0.5, 1)
        img_aug = seq_fog.augment_image(img_aug)
        img_aug = cv2.addWeighted(img_aug, 0.6, img, 0.4, 1)
    return img_aug


imgpath = "/data/share/kitti_data_semantics/images/training/"
# maskpath = "/data/share/kitti_data_semantics/annotations/training/"
img_list = glob(imgpath + '*.png')
for i, path in enumerate(img_list):
    img = cv2.imread(path)
    img2 = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    plt.imshow(img2)
    plt.show()
    maskpath = path.replace('images', 'annotations').replace('rgb_anon', 'gt_labelTrainIds')
    mask = cv2.imread(maskpath, 0)
    img_aug = GenWeather(img, mask)
    img_aug = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)
    plt.suptitle('{}'.format(i))
    plt.imshow(img_aug)
    plt.show()

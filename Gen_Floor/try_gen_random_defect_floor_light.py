# coding=gbk

import os
from cv2 import inpaint
import glob

import numpy as np
import cv2
import json

from shapely.geometry import MultiPoint
from descartes.patch import PolygonPatch
from pycocotools import mask as coco_mask
from cv2.ximgproc import *
from pylab import plt

data_fold = "D:/python_pro/Datasets/anno_data_v220415/half_segms/"
MIN_POINTS = 5
MAX_POINTS = 20
label_map = {'floor': 1, 'light': 2, 'defect': 3}

image_paths = glob.glob(data_fold + '*.bmp') + glob.glob(data_fold + '*.jpg')
for image_path in image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ori_image = image.copy()
    height, width = image.shape

    json_path = image_path.replace('.bmp', '.json').replace('.jpg', '.json')
    with open(json_path, 'r') as f:
        labels = json.load(f)
    labels = labels['shapes']

    label_mask = np.zeros_like(image)
    for label in labels:
        mask = []
        for m in label['points']:
            mask.append(m[0])
            mask.append(m[1])
        mask = coco_mask.frPyObjects([mask], height, width)
        mask = coco_mask.decode(mask)
        if len(mask.shape) == 3:
            mask = np.sum(mask, axis=2)

        mask[label_mask != 0] = 0
        cat = label_map[label['label']]
        label_mask[mask != 0] = cat

    # random floor
    floor_map_ratio = np.random.uniform() * 1.0 + 0.5
    image = image.astype(np.float32)
    image[label_mask == label_map['floor']] *= floor_map_ratio
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype(np.uint8)

    # inpaint the non-floor  修复缺陷？
    inpaint_image = image.copy()
    inpaint_mask = (label_mask != label_map['floor']).astype(np.uint8) * 255
    if (inpaint_mask != 0).sum() != 0:  # inpaintmask中有0像素点
        # gf to mask
        inpaint_mask = cv2.ximgproc.guidedFilter(guide=inpaint_image, src=inpaint_mask, radius=20, eps=100, dDepth=-1)
        inpaint_mask[inpaint_mask != 0] = 255
        inpaint_image = cv2.inpaint(inpaint_image, inpaint_mask, 5, cv2.INPAINT_NS)

    # get random points
    defect_image = image.copy()
    defect_mask = np.zeros_like(image)
    light_region_mask = (label_mask == label_map['light']).astype(np.uint8) * 255
    if (light_region_mask != 0).sum() != 0:
        light_region_locs = np.where(label_mask == label_map['light'])  # 返回label_mask等于2的位置
        light_min_row = min(light_region_locs[0])
        light_max_row = max(light_region_locs[0])
        light_min_col = min(light_region_locs[1])
        light_max_col = max(light_region_locs[1])

        floor_region_locs = np.where(label_mask == label_map['floor'])
        floor_min_row = min(floor_region_locs[0])
        floor_max_row = max(floor_region_locs[0])
        floor_min_col = min(floor_region_locs[1])
        floor_max_col = max(floor_region_locs[1])

        # # extend light regions
        # light_min_row = max(light_min_row * 2 - light_max_row, floor_min_row)
        # light_max_row = min(light_max_row * 2 - light_min_row, floor_max_row)
        # randomly extend light regions
        random_up_ratio = np.random.uniform()
        random_down_ratio = np.random.uniform()
        random_left_ratio = np.random.uniform()
        random_len_ratio = np.random.uniform() * 0.5 + 0.5
        light_min_row = max(light_min_row - random_up_ratio * (light_max_row - light_min_row), floor_min_row)
        light_max_row = min(light_max_row + random_down_ratio * (light_max_row - light_min_row), floor_max_row)
        light_min_col = max(light_min_col + random_left_ratio * (light_max_col - light_min_col), floor_min_col)
        light_len_col = random_len_ratio * (light_max_col - light_min_col)
        light_max_col = min(light_min_col + light_len_col, floor_max_col)
        light_min_row = int(light_min_row)
        light_max_row = int(light_max_row)
        light_min_col = int(light_min_col)
        light_max_col = int(light_max_col)

        # random polygon points
        num_points = np.random.choice(range(MIN_POINTS, MAX_POINTS))
        center_y = np.random.choice(range(light_min_row, light_max_row))
        center_x = np.random.choice(range(light_min_col, light_max_col))
        start_y = np.random.choice(range(light_min_row, light_max_row))
        start_x = np.random.choice(range(light_min_col, light_max_col))
        pre_y = start_y
        pre_x = start_x
        for pi in range(num_points):
            if pi == (num_points - 1):
                cur_y = start_y
                cur_x = start_x
            else:
                cur_y = np.random.choice(range(light_min_row, light_max_row))
                cur_x = np.random.choice(range(light_min_col, light_max_col))

            mask = [center_x, center_y, pre_x, pre_y, cur_x, cur_y]
            mask = coco_mask.frPyObjects([mask], height, width)
            mask = coco_mask.decode(mask)
            if len(mask.shape) == 3:
                mask = np.sum(mask, axis=2)

            mask[defect_mask != 0] = 0
            defect_mask[mask != 0] = 1

            pre_y = cur_y
            pre_x = cur_x

        # convert mask
        cat = label_map['defect']
        label_mask[defect_mask != 0] = cat

        # convert image
        defect_image = defect_image.astype(np.float32)
        defect_image[defect_mask != 0] = inpaint_image[defect_mask != 0] + np.random.uniform() * 50 - 10
        defect_image[defect_image > 255] = 255
        defect_image = defect_image.astype(np.uint8)

    # random light
    light_image = defect_image.copy()
    light_map_ratio = np.random.uniform() * 1.0 + 0.5
    light_image = light_image.astype(np.float32)
    light_image *= light_map_ratio
    light_image[light_image > 255] = 255
    light_image[light_image < 0] = 0
    light_image = light_image.astype(np.uint8)

    plt.subplot(231)
    plt.imshow(ori_image, cmap="gray")
    plt.subplot(232)
    plt.imshow(label_mask)
    plt.subplot(233)
    plt.imshow(image, cmap="gray")
    plt.subplot(234)
    plt.imshow(defect_image, cmap="gray")
    plt.subplot(235)
    plt.imshow(light_image, cmap="gray")
    plt.show()

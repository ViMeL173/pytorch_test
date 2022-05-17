# coding=gbk

import numpy as np
import math


def LBP(src):  # 3*3����
    '''
    :param src:�Ҷ�ͼ��
    :return:
    '''
    height = src.shape[0]
    width = src.shape[1]
    dst = src.copy()
    lbp_value = np.zeros((1, 8), dtype=np.uint8)
    neighbours = np.zeros((1, 8), dtype=np.uint8)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            neighbours[0, 0] = src[y - 1, x - 1]
            neighbours[0, 1] = src[y - 1, x]
            neighbours[0, 2] = src[y - 1, x + 1]
            neighbours[0, 3] = src[y, x - 1]
            neighbours[0, 4] = src[y, x + 1]
            neighbours[0, 5] = src[y + 1, x - 1]
            neighbours[0, 6] = src[y + 1, x]
            neighbours[0, 7] = src[y + 1, x + 1]

            center = src[y, x]

            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                  + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 0] * 128

            dst[y, x] = lbp

    return dst


def circular_LBP(src, radius, n_points):  # Բ������
    height = src.shape[0]
    width = src.shape[1]
    dst = src.copy()
    src.astype(dtype=np.float32)
    dst.astype(dtype=np.float32)

    neighbours = np.zeros((1, n_points), dtype=np.uint8)
    lbp_value = np.zeros((1, n_points), dtype=np.uint8)
    for x in range(radius, width - radius - 1):
        for y in range(radius, height - radius - 1):
            lbp = 0.
            # �ȼ��㹲n_points�����Ӧ������ֵ��ʹ��˫���Բ�ֵ��
            for n in range(n_points):
                theta = float(2 * np.pi * n) / n_points
                x_n = x + radius * np.cos(theta)
                y_n = y - radius * np.sin(theta)

                # ����ȡ��
                x1 = int(math.floor(x_n))
                y1 = int(math.floor(y_n))
                # ����ȡ��
                x2 = int(math.ceil(x_n))
                y2 = int(math.ceil(y_n))

                # ������ӳ�䵽0-1֮��
                tx = np.abs(x - x1)
                ty = np.abs(y - y1)

                # ����0-1֮���x��y��Ȩ�ؼ��㹫ʽ����Ȩ��
                w1 = (1 - tx) * (1 - ty)
                w2 = tx * (1 - ty)
                w3 = (1 - tx) * ty
                w4 = tx * ty

                # ����˫���Բ�ֵ��ʽ�����k��������ĻҶ�ֵ
                neighbour = src[y1, x1] * w1 + src[y2, x1] * w2 + src[y1, x2] * w3 + src[y2, x2] * w4

                neighbours[0, n] = neighbour

            center = src[y, x]

            for n in range(n_points):
                if neighbours[0, n] > center:
                    lbp_value[0, n] = 1
                else:
                    lbp_value[0, n] = 0

            for n in range(n_points):
                lbp += lbp_value[0, n] * 2 ** n

            # ת����0-255�ĻҶȿռ䣬����n_points=16λʱ����ᳬ�������Χ���Ըý����һ��
            dst[y, x] = int(lbp / (2 ** n_points - 1) * 255)

    return dst


# ��ת����
def value_rotation(num):
    value_list = np.zeros((8), np.uint8)
    temp = int(num)
    value_list[0] = temp
    for i in range(7):
        temp = ((temp << 1) | (temp / 128)) % 256
        value_list[i + 1] = temp
    return np.min(value_list)


def rotation_invariant_LBP(src):
    height = src.shape[0]
    width = src.shape[1]
    dst = src.copy()

    lbp_value = np.zeros((1, 8), dtype=np.uint8)
    neighbours = np.zeros((1, 8), dtype=np.uint8)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            neighbours[0, 0] = src[y - 1, x - 1]
            neighbours[0, 1] = src[y - 1, x]
            neighbours[0, 2] = src[y - 1, x + 1]
            neighbours[0, 3] = src[y, x - 1]
            neighbours[0, 4] = src[y, x + 1]
            neighbours[0, 5] = src[y + 1, x - 1]
            neighbours[0, 6] = src[y + 1, x]
            neighbours[0, 7] = src[y + 1, x + 1]

            center = src[y, x]

            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                  + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 0] * 128

            # ��ת����ֵ
            dst[y, x] = value_rotation(lbp)

    return dst


# ����ģʽLBP
def getHopCnt(num):
    '''
    :param num:8λ����������0-255
    :return:
    '''
    if num > 255:
        num = 255
    elif num < 0:
        num = 0

    num_b = bin(num)
    num_b = str(num_b)[2:]

    # ��0
    if len(num_b) < 8:
        temp = []
        for i in range(8 - len(num_b)):
            temp.append('0')
        temp.extend(num_b)
        num_b = temp

    cnt = 0
    for i in range(8):
        if i == 0:
            former = num_b[-1]
        else:
            former = num_b[i - 1]
        if former == num_b[i]:
            pass
        else:
            cnt += 1

    return cnt


def img_max_min_normalization(src, min=0, max=255):
    height = src.shape[0]
    width = src.shape[1]
    if len(src.shape) > 2:
        channel = src.shape[2]
    else:
        channel = 1

    src_min = np.min(src)
    src_max = np.max(src)

    if channel == 1:
        dst = np.zeros([height, width], dtype=np.float32)
        for h in range(height):
            for w in range(width):
                dst[h, w] = float(src[h, w] - src_min) / float(src_max - src_min) * (max - min) + min
    else:
        dst = np.zeros([height, width, channel], dtype=np.float32)
        for c in range(channel):
            for h in range(height):
                for w in range(width):
                    dst[h, w, c] = float(src[h, w, c] - src_min) / float(src_max - src_min) * (max - min) + min

    return dst


def uniform_LBP(src, norm=True):
    '''
    :param src:ԭʼͼ��
    :param norm:�Ƿ�����һ������0-255���ĻҶȿռ�
    :return:
    '''
    table = np.zeros((256), dtype=np.uint8)
    temp = 1
    for i in range(256):
        if getHopCnt(i) <= 2:
            table[i] = temp
            temp += 1

    height = src.shape[0]
    width = src.shape[1]
    dst = np.zeros([height, width], dtype=np.uint8)
    dst = src.copy()

    lbp_value = np.zeros((1, 8), dtype=np.uint8)
    neighbours = np.zeros((1, 8), dtype=np.uint8)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            neighbours[0, 0] = src[y - 1, x - 1]
            neighbours[0, 1] = src[y - 1, x]
            neighbours[0, 2] = src[y - 1, x + 1]
            neighbours[0, 3] = src[y, x - 1]
            neighbours[0, 4] = src[y, x + 1]
            neighbours[0, 5] = src[y + 1, x - 1]
            neighbours[0, 6] = src[y + 1, x]
            neighbours[0, 7] = src[y + 1, x + 1]

            center = src[y, x]

            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                  + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 0] * 128

            dst[y, x] = table[lbp]

    if norm is True:
        return img_max_min_normalization(dst)
    else:
        return dst


def rotation_invariant_uniform_LBP(src):  # ����+��ת����
    table = np.zeros((256), dtype=np.uint8)
    temp = 1
    for i in range(256):
        if getHopCnt(i) <= 2:
            table[i] = temp
            temp += 1

    height = src.shape[0]
    width = src.shape[1]
    dst = np.zeros([height, width], dtype=np.uint8)
    dst = src.copy()

    lbp_value = np.zeros((1, 8), dtype=np.uint8)
    neighbours = np.zeros((1, 8), dtype=np.uint8)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            neighbours[0, 0] = src[y - 1, x - 1]
            neighbours[0, 1] = src[y - 1, x]
            neighbours[0, 2] = src[y - 1, x + 1]
            neighbours[0, 3] = src[y, x - 1]
            neighbours[0, 4] = src[y, x + 1]
            neighbours[0, 5] = src[y + 1, x - 1]
            neighbours[0, 6] = src[y + 1, x]
            neighbours[0, 7] = src[y + 1, x + 1]

            center = src[y, x]

            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                  + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 0] * 128

            dst[y, x] = table[lbp]

    dst = img_max_min_normalization(dst)
    for x in range(width):
        for y in range(height):
            dst[y, x] = value_rotation(dst[y, x])

    return dst

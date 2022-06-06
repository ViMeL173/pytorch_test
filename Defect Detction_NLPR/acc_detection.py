# coding=gbk

import cv2
import matplotlib.pyplot as plt
import os
import glob


def imshow(img, n=0):  # n=0为错误检测，n=1为漏检
    if n == 0:
        plt.suptitle("False detection")
    if n == 1:
        plt.suptitle("Missing")
    else:
        pass
    plt.imshow(img, "gray")
    plt.savefig("img_cannot_detect/" + imgname)
    plt.show()


# flag = 0

# path = "D:/python_pro/python_proj/yolov5-master/data/defect_data/images/test/"
# imgname = "cam-0_20220227_143205-136.jpg"
# imgpath = path + imgname
# basename = os.path.basename(imgpath)
# print(basename,type(basename))
# txtpath = imgpath.replace(".jpg", ".txt").replace("images", "labels")
# img = cv2.imread(imgpath, 0)
#
# # cv2.imwrite("img_cannot_detect/" + imgname, img)
# with open(txtpath) as f:
#     lines = f.readlines()
#     for line in lines:
#         if line[0] == '2':
#             flag = 1
# imshow(img, flag)

imgpaths = glob.glob("img_cannot_detect/*.jpg")
for imgpath in imgpaths:
    imgname = os.path.basename(imgpath)
    img = cv2.imread("D:/python_pro/python_proj/yolov5-master/data/defect_data/images/test/" + imgname, 0)
    cv2.imwrite("miss_detection_ori_img/" + imgname, img)

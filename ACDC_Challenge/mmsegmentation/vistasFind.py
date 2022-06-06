import glob
import cv2
import matplotlib.pyplot as plt


n = 53
num = 10

count = 0

imgpath = '/data/share/MapillaryVistas/training/instances/'
# labelpath = "D:\python_pro\python_proj\pytorch_test\ACDC_Challenge\mmsegmentation\data/acdc_challenge/annotations/training\GOPR0122_frame_000161_gt_labelTrainIds.png"
labelpath = glob.glob(imgpath + '*.png')
for i in labelpath:
    img = cv2.imread(i, 0)
    if n in img:
        img[img == n] = 255
        plt.imshow(img, cmap=plt.cm.gray)
        plt.show()
        img2path = i.replace('png', 'jpg').replace('instances', 'images')
        img2 = cv2.imread(img2path)
        plt.imshow(img2)
        plt.show()
        print(i)
        count += 1
        if count == num:
            break

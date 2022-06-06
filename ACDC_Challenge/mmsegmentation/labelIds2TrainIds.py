import os, glob, cv2
from tqdm import tqdm
import shutil


def decodeImg(img):
    img[img == 0] = 255
    img[img == 1] = 255
    img[img == 2] = 255
    img[img == 3] = 255
    img[img == 4] = 255
    img[img == 5] = 255
    img[img == 6] = 255
    img[img == 9] = 255
    img[img == 10] = 255
    img[img == 14] = 255
    img[img == 15] = 255
    img[img == 16] = 255
    img[img == 18] = 255
    img[img == 29] = 255
    img[img == 30] = 255

    img[img == 7] = 0  # road
    img[img == 8] = 1
    img[img == 11] = 2
    img[img == 12] = 3
    img[img == 13] = 4
    img[img == 17] = 5
    img[img == 19] = 6
    img[img == 20] = 7
    img[img == 21] = 8
    img[img == 22] = 9
    img[img == 23] = 10
    img[img == 24] = 11
    img[img == 25] = 12
    img[img == 26] = 13
    img[img == 27] = 14
    img[img == 28] = 15
    img[img == 31] = 16
    img[img == 32] = 17
    img[img == 33] = 18

    return img


city_dir = 'E:/Deep_Learning/Datasets/kitti_data_semantics/'

target = ['training/', 'testing/']
for t in target:
    ori_label_list = []
    path = city_dir + t + 'semantic/'
    target_path = city_dir + 'annotations/' + t
    target_path2 = city_dir + 'images/' + t
    label_list = glob.glob(path + '*.png')  # _gtFine_labelIds
    for label in tqdm(label_list, desc='changing', mininterval=0.1):
        imgpath = label.replace('semantic\\', 'image_2\\')
        img = cv2.imread(label, 0)
        basename = os.path.basename(label)
        img_trainid = decodeImg(img)
        imgname = basename.replace('.', '_gt_labelTrainIds.')
        cv2.imwrite(target_path + imgname, img_trainid)

        basename2 = os.path.basename(imgpath)
        imgname2 = basename2.replace('.', '_rgb_anon.')
        shutil.copyfile(imgpath, target_path2 + imgname2)

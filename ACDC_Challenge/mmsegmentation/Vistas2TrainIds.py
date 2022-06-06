import os, glob, cv2
from tqdm import tqdm
import shutil


def decodeImg(img):
    img[img == 13] = 0  # road
    img[img == 24] = 0  # road
    img[img == 41] = 0  # road
    img[img == 7] = 0  # road
    img[img == 14] = 0  # road
    img[img == 23] = 0  # road
    img[img == 2] = 1
    img[img == 15] = 1
    img[img == 9] = 1
    img[img == 11] = 1
    img[img == 17] = 2
    img[img == 6] = 3
    img[img == 3] = 4
    img[img == 45] = 5
    img[img == 46] = 5
    img[img == 47] = 5
    img[img == 48] = 6
    img[img == 50] = 7
    img[img == 30] = 8
    img[img == 29] = 9
    img[img == 25] = 9
    img[img == 27] = 10
    img[img == 19] = 11
    img[img == 20] = 12
    img[img == 21] = 12
    img[img == 22] = 12
    img[img == 55] = 13
    img[img == 61] = 14
    img[img == 54] = 15
    img[img == 58] = 16
    img[img == 57] = 17
    img[img == 52] = 18

    img[img > 18] = 255

    return img


def del_uesless(img):
    flag = 0
    if 28 in img:
        flag = 1
    if 31 in img:
        flag = 1
    if 36 in img:
        flag = 1
    if 43 in img:
        flag = 1
    if 53 in img:
        flag = 1
    return flag


city_dir = '/data/share/MapillaryVistas/'

n = 0

target = ['training/', 'validation/']
for t in target:
    ori_label_list = []
    ori_img_list = []
    path_label = city_dir + t + 'instances/'
    path_img = city_dir + t + 'images/'
    target_path = city_dir + 'annotations/' + t
    target_path2 = city_dir + 'images/' + t
    city_lists = os.listdir(path_label)
    for city_list in tqdm(city_lists, desc='changing', mininterval=0.1):
        label_path = path_label + city_list
        img_path = path_img + city_list.replace('.png', '.jpg')
        img = cv2.imread(label_path, 0)
        if del_uesless(img):
            print("del")
            continue
        else:
            img_trainid = decodeImg(img)

            labelname = 'Vistas_' + '{}'.format(n) + '_gt_labelTrainIds.png'
            cv2.imwrite(target_path + labelname, img_trainid)

            imgname = 'Vistas_' + '{}'.format(n) + '_rgb_anon.png'
            shutil.copyfile(img_path, target_path2 + imgname)
            n += 1

# for t in target:
#     ori_img_list = []
#     path2 = city_dir + t
#     target_path2 = city_dir + 'images/' + t
#     city_img_lists = os.listdir(path2)
#     for city_img_list in city_img_lists:
#         label_list2 = glob.glob(path2 + city_img_list + '/*.png')
#         for label2 in label_list2:
#             ori_img_list.append(label2)
#     for ori_img in tqdm(ori_img_list, desc=t + ':img-copy', mininterval=0.1):
#         basename2 = os.path.basename(ori_img)
#         imgname2 = basename2.replace('_leftImg8bit', '_rgb_anon')
#         shutil.copyfile(ori_img, target_path2 + imgname2)

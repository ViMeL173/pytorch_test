import os, glob, cv2
from tqdm import tqdm
import shutil
import argparse


def myrename(old, filename):
    if 'image' in filename:
        newone = old.replace('.jpg', '_rgb_anon.png')
    elif 'label' in filename:
        newone = old.replace('train_id', 'gt_labelTrainIds')
    else:
        newone = old
    return newone


def getdir(filename):
    if 'image' in filename:
        dirpath = 'images/training'
    elif 'segm' in filename:
        dirpath = 'annotations/training'
    else:
        dirpath = filename
    return dirpath


def parse_args():
    parser = argparse.ArgumentParser(description='Rename the img')
    parser.add_argument('target_path', help='path of renamed file')
    parser.add_argument('--option', default='rename', help='choose rename or copy')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    option = args.option
    target_path = args.target_path
    in_paths = ['images', 'labels']
    in_path2 = ['train', 'val']

    for ip in in_paths:
        for ip2 in in_path2:
            path = os.path.join(target_path, ip)
            img_or_gt = os.path.basename(path)
            path = os.path.join(path, ip2)
            imggt_list = os.listdir(path)
            imggt_paths = [os.path.join(path, imggt) for imggt in imggt_list]
            copy_path = getdir(img_or_gt)
            copy_path = os.path.join(target_path, copy_path)
            for imggt_path in tqdm(imggt_paths, desc=ip + '/' + ip2 + "Rename", mininterval=0.5):
                oldname = os.path.basename(imggt_path)
                newname = myrename(oldname, img_or_gt)
                newname_path = imggt_path.replace(oldname, newname)
                cpath = os.path.join(copy_path, newname)
                if option == 'copy':
                    shutil.copyfile(imggt_path, cpath)
                else:
                    os.rename(imggt_path, newname_path)

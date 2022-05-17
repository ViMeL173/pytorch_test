# coding=gbk


from torch.utils.data import Dataset
from PIL import Image


# 读取数据集label像素信息并进行标记


# 读取大类分组并标记
class DAGM2007(Dataset):
    def __init__(self, train=True, transform=None, update_dataset=False, mode='Class'):

        dataset_path = 'D:/python_pro/python_proj/pytorch_test/dataset/DAGM/'

        if train:
            self.dataset_type = 'Train'
        else:
            self.dataset_type = 'Test'

        if update_dataset:
            pass
        self.transform = transform
        self.sample_list = list()
        self.mode = mode
        if self.mode != 'Class' or 'Detection':
            self.mode = 'Class'
        if self.mode == 'Class':
            for i in range(1, 7):
                with open(dataset_path + 'Class{}/'.format(i) + self.dataset_type + '/Label/labels.txt') as f:  # 锁定目标文件
                    lines = f.readlines()
                    img_path = dataset_path + 'Class{}/'.format(i) + self.dataset_type + '/' + '\t'  # 图像文件前置路径
                    for line in lines:
                        if line == '1\n':
                            continue  # 去除第一行的“1”
                        self.sample_list.append(
                            img_path + line.strip() + '\t{}'.format(i-1))  # 读取txt内容并保存于list中(i为class标签)
        elif self.mode == 'Detection':
            pass

    def __getitem__(self, index):
        item = self.sample_list[index]
        img = Image.open(item.split()[0] + item.split()[3]).convert('L')
        if self.transform is not None:
            img = self.transform(img)
        label = int(item.split()[6])
        return img, label

    def __len__(self):
        return len(self.sample_list)

    def get_name(self, index):
        item = self.sample_list[index]
        return item.split()[3]

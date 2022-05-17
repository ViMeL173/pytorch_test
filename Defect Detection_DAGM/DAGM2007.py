# coding=gbk


from torch.utils.data import Dataset
from PIL import Image


# ��ȡ���ݼ�label������Ϣ�����б��


# ��ȡ������鲢���
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
                with open(dataset_path + 'Class{}/'.format(i) + self.dataset_type + '/Label/labels.txt') as f:  # ����Ŀ���ļ�
                    lines = f.readlines()
                    img_path = dataset_path + 'Class{}/'.format(i) + self.dataset_type + '/' + '\t'  # ͼ���ļ�ǰ��·��
                    for line in lines:
                        if line == '1\n':
                            continue  # ȥ����һ�еġ�1��
                        self.sample_list.append(
                            img_path + line.strip() + '\t{}'.format(i-1))  # ��ȡtxt���ݲ�������list��(iΪclass��ǩ)
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

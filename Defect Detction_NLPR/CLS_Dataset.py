# coding=gbk

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# 数据集处理与标记

class CLSdataset(Dataset):
    def __init__(self, train=True, transform=None, mode=0):
        '''
        :param mode: 0：不做处理
                     1：裁剪拼接
                     2：填充策略
                     3：裁剪填充拼接
        '''

        dataset_path = "/data/share/Oppein/first_process/cls_data/"

        if train:
            self.dataset_type = 'train.txt'
        else:
            self.dataset_type = 'val.txt'
        self.mode = mode
        self.transform = transform
        self.sample_list = list()
        with open(dataset_path + self.dataset_type) as f:  # 锁定目标文件
            lines = f.readlines()
            img_path = dataset_path + 'images' + '/'  # 图像文件前置路径
            for line in lines:
                self.sample_list.append(
                    img_path + line.strip())  # 读取txt内容并保存于list中(i为class标签)

    def __getitem__(self, index):
        item = self.sample_list[index]
        img = Image.open(item.split()[0]).convert('L')

        if self.mode == 0:
            img2 = img

        elif self.mode == 1:
            p = []
            for i in range(4):
                p.append(img.crop((i * 612, 0, 612 * (i + 1), img.height)))
            img2 = Image.new('L', (612, img.height * 4))
            for i in range(4):
                img2.paste(p[i], (0, img.height * i, 612, img.height * (i + 1)))

        elif self.mode == 2:
            img1 = transforms.Resize([img.height, 3 * img.height])(img)
            img2 = transforms.Pad(padding=(0, img.height))(img1)

        elif self.mode == 3:
            p = []
            for i in range(3):
                p.append(transforms.Pad(padding=(0, 4))(img.crop((i * 816, 0, 816 * (i + 1), img.height))))
            img2 = Image.new('L', (816, (img.height * 3 + 8 * 3)))
            for i in range(3):
                img2.paste(p[i], (0, (img.height + 8) * i, 816, (img.height + 8) * (i + 1)))

        else:
            img2 = img

        if self.transform is not None:
            img2 = self.transform(img2)
        label = int(item.split()[1])
        return img2, label

    def __len__(self):
        return len(self.sample_list)

    def get_name(self, index):
        item = self.sample_list[index]
        return item.split()[0]


def Normalize_CLS(img_size, mode):
    '''
    This function is for transforms.Normalize
    #0.4350961744785309 0.23526135087013245
    '''
    transform = transforms.Compose(
        [transforms.Resize([img_size, img_size]), transforms.ToTensor()])
    cls_train = CLSdataset(transform=transform, train=True, mode=mode)
    cls_test = CLSdataset(transform=transform, train=False, mode=mode)
    train_loader = DataLoader(cls_train, shuffle=True, batch_size=4711)
    test_loader = DataLoader(cls_test, shuffle=True, batch_size=866)
    for idx, data in enumerate(train_loader, 0):
        ipt, tg = data
        x = ipt.view(-1, img_size * img_size)
        x_std = x.std().item()
        x_mean = x.mean().item()
    for data in test_loader:
        ipt1, tg1 = data
        y = ipt1.view(-1, img_size * img_size)
        y_std = y.std().item()
        y_mean = y.mean().item()

    return x_mean, x_std, y_mean, y_std

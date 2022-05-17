# coding=gbk

from DAGM2007 import DAGM2007
from torchvision import transforms
from torch.utils.data import DataLoader


def Normalize_DAMG(Mode='class'):
    '''

    :param Mode: class:background classification
                 detection:detect detection
    :return: x_mean and x_std
    This function is for transforms.Normalize
    '''
    if Mode == 'class':
        batch_size = 3450
        transform = transforms.Compose(
            [transforms.Resize([128, 128]), transforms.ToTensor()])
        dagm_train = DAGM2007(transform=transform, train=True)
        train_loader = DataLoader(dagm_train, shuffle=True, batch_size=batch_size)
        for idx, data in enumerate(train_loader, 0):
            ipt, tg = data
            x = ipt.view(-1, 128 * 128)
            x_std = x.std().item()
            x_mean = x.mean().item()
    if Mode == 'detection':
        pass

    else:
        x_std = x_mean = 0

    return x_mean, x_std

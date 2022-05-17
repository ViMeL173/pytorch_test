# coding=gbk

from torch import nn
def vgg16(batch_norm=False) -> nn.ModuleList:
    """ ���� vgg16 ģ��
    batch_norm: bool(�Ƿ��ھ��������������һ����)
    """
    layers = []
    in_channels = 3
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256,
           256, 'C', 512, 512, 512, 'M', 512, 512, 512]
    # ������Ԥ���������ʼ��

    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(2, 2))
        elif v == 'C':
            layers.append(nn.MaxPool2d(2, 2, ceil_mode=True))
        else:
            conv = nn.Conv2d(in_channels, v, 3, padding=1)

            # �����Ҫ����һ���Ĳ��������һ������һ����
            if batch_norm:
                layers.extend([conv, nn.BatchNorm2d(v), nn.ReLU(True)])
            else:
                layers.extend([conv, nn.ReLU(True)])

            in_channels = v

    # ��ԭʼ�� fc6��fc7 ȫ���Ӳ��滻Ϊ�����
    layers.extend([
        nn.MaxPool2d(3, 1, 1),
        nn.Conv2d(512, 1024, 3, padding=6, dilation=6),  # conv6 ʹ�ÿն�������Ӹ���Ұ
        nn.ReLU(True),
        nn.Conv2d(1024, 1024, 1),                        # conv7
        nn.ReLU(True)
    ])

    layers = nn.ModuleList(layers)
    return layers

print(vgg16())
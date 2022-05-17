# coding=gbk

from torch import nn
def vgg16(batch_norm=False) -> nn.ModuleList:
    """ 创建 vgg16 模型
    batch_norm: bool(是否在卷积层后面添加批归一化层)
    """
    layers = []
    in_channels = 3
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256,
           256, 'C', 512, 512, 512, 'M', 512, 512, 512]
    # 网络框架预览，方便初始化

    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(2, 2))
        elif v == 'C':
            layers.append(nn.MaxPool2d(2, 2, ceil_mode=True))
        else:
            conv = nn.Conv2d(in_channels, v, 3, padding=1)

            # 如果需要批归一化的操作就添加一个批归一化层
            if batch_norm:
                layers.extend([conv, nn.BatchNorm2d(v), nn.ReLU(True)])
            else:
                layers.extend([conv, nn.ReLU(True)])

            in_channels = v

    # 将原始的 fc6、fc7 全连接层替换为卷积层
    layers.extend([
        nn.MaxPool2d(3, 1, 1),
        nn.Conv2d(512, 1024, 3, padding=6, dilation=6),  # conv6 使用空洞卷积增加感受野
        nn.ReLU(True),
        nn.Conv2d(1024, 1024, 1),                        # conv7
        nn.ReLU(True)
    ])

    layers = nn.ModuleList(layers)
    return layers

print(vgg16())
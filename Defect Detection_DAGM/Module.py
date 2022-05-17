from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Net(nn.Module):
    def __init__(self, m=0):
        super(Net, self).__init__()
        self.Mode = m  # 设置模式：背景分类or缺陷检测

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # w = nn.init.trunc_normal_(torch.randn(32, 1, 3, 3), mean=0, std=2 / 32)  # 权重参数更具截断随机正态分布初始化
        # self.conv1.weight = Parameter(w)  # 更新卷积层的权重

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # w = nn.init.trunc_normal_(torch.randn(64, 32, 3, 3), mean=0, std=2 / (32 * 64))
        # self.conv2_1.weight = Parameter(w)  # 更新卷积层的权重

        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # w = nn.init.trunc_normal_(torch.randn(64, 64, 3, 3), mean=0, std=2 / (64 * 64))
        # self.conv2_2.weight = Parameter(w)  # 更新卷积层的权重

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # w = nn.init.trunc_normal_(torch.randn(128, 64, 3, 3), mean=0, std=2 / (64 * 128))
        # self.conv3_1.weight = Parameter(w)  # 更新卷积层的权重

        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # w = nn.init.trunc_normal_(torch.randn(128, 128, 3, 3), mean=0, std=2 / (128 * 128))
        # self.conv3_2.weight = Parameter(w)  # 更新卷积层的权重

        self.pooling = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32768, 1024)
        # w = nn.init.trunc_normal_(torch.randn(1024, 32768), mean=0, std=2 / (32768 * 1024))
        # self.fc1.weight = Parameter(w)

        self.fc2 = nn.Linear(1024, 1024)
        # w = nn.init.trunc_normal_(torch.randn(1024, 1024), mean=0, std=2 / (1024 * 1024))
        # self.fc2.weight = Parameter(w)

        self.fc_final6 = nn.Linear(1024, 6)  # FC到6维是大分类(背景)
        # w = nn.init.trunc_normal_(torch.randn(6, 1024), mean=0, std=2 / (6 * 1024))
        # self.fc_final6.weight = Parameter(w)

        self.fc_final2 = nn.Linear(1024, 2)  # FC到2维是分类是否有缺陷
        # w = nn.init.trunc_normal_(torch.randn(2, 1024), mean=0, std=2 / (2 * 1024))
        # self.fc_final2.weight = Parameter(w)

    def forward(self, x):
        batch_size = x.size(0)  # 得到batch size,文章中为50
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2_2(F.relu(self.conv2_1(x)))))
        x = self.pooling(F.relu(self.conv3_2(F.relu(self.conv3_1(x)))))
        x = x.view(batch_size, -1)
        x = self.fc2(self.fc1(x))
        x = F.dropout(x, p=0.5)  # Dropout策略，概率为0.5
        if self.Mode == 0:  # 选择网络模式
            x = self.fc_final6(x)
        elif self.Mode == 1:
            x = self.fc_final2(x)
        else:
            x = self.fc_final6(x)
        return x

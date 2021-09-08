import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

n = 100


def mymodify(n):
    if n < 0.5 and n >= 0:
        m = 0
    elif n > 0.5 and n <= 1:
        m = 1
    elif n == 0.5:
        m = 0.5
    else:
        m = None
    return m


# =================================准备数据集================================== #

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)  # 以逗号分隔读取数据
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('diabetes1.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)  # 加载器


# =================================准备模型================================== #

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)  # 输入8维，输出6维
        self.linear2 = torch.nn.Linear(6, 4)  # 输入6维，输出4维
        self.linear3 = torch.nn.Linear(4, 1)  # 输入4维，输出1维
        self.sigmoid = torch.nn.Sigmoid()  # 这是一个模块（层）,激活函数

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

# =================================调用API初始化模块================================== #

criterion = torch.nn.BCELoss(size_average=False)  # BCE(二值交叉熵)损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 优化器（用于更新w）

# 不使用datase时
# for epoch in range(n):
#     y_pred = model(x_data)
#     loss = criterion(y_pred, y_data)
#     print(epoch, loss.item())
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()


if __name__ == '__main__':
    # =================================进行训练循环================================== #

    for epoch in range(n):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # =================================测试结果================================== #
    xy_test = np.loadtxt('diabetes_test.csv', delimiter=',', dtype=np.float32)
    x_test = torch.from_numpy(xy_test[:, :-1])
    y_test = torch.from_numpy(xy_test[:, [-1]])

    for i in range(xy_test.shape[0]):
        y_hat = model(x_test[i])
        print('y_hat = ', mymodify(y_hat.item()), 'y = ', y_test[i].item())

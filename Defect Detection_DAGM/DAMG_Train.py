# coding=gbk

from DAMG_Normalize import Normalize_DAMG
from Module import Net
from DAGM2007 import DAGM2007
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# 定义batch size，求出训练集中值和均值用于transform
batch_size = 50
mean1, std1 = Normalize_DAMG()
transform = transforms.Compose(
    [transforms.Resize([128, 128]), transforms.ToTensor(), transforms.Normalize((0.46), (0.17,))])

dagm_train = DAGM2007(transform=transform, train=True)
train_loader = DataLoader(dagm_train, shuffle=True, batch_size=batch_size)
dagm_test = DAGM2007(transform=transform, train=False)
test_loader = DataLoader(dagm_test, shuffle=False, batch_size=batch_size)

model = Net(m=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  # GPU训练

criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失 log_softmax+NLL_loss
LR = 0.001  # 初始学习率
# 优化器初始化 L2正则化（正则化系数=5*10^-5）、学习率、动量系数=0.9
optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=5e-5, momentum=0.9)  # weight_decay=5e-5,momentum=0.9

loss_list = []
acc_list = []


def Learning_rate_attenuation(lr, idx):  # 学习率衰减函数，衰减系数为0.012
    return lr / (1 + 0.012 * idx)


def train(epoch):
    running_loss = 0.0
    global loss_list
    counter = 0
    for idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        # for p in optimizer.param_groups:
        #     p['lr'] = Learning_rate_attenuation(p['lr'], idx)
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if idx % 30 == 29:  # 每30论输出一个loss
            print('[%d, %5d] loss: %.3f' % (epoch + 1, idx + 1, running_loss / 30))
            loss_list.append(running_loss / 30)
            counter += counter
            running_loss = 0.0
    return counter


def test():
    correct = 0
    total = 0
    global acc_list
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # 取一行中最大值的下标号
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 计算正确的个数
    print('正确率: %d %%' % (100 * correct / total))
    acc_list.append(100 * correct / total)


def draw_loss(c, e):
    global loss_list
    global acc_list
    plt.plot(range(c), loss_list, color='r')
    plt.plot(range(e), acc_list, color='r')
    plt.show()


if __name__ == '__main__':
    c = 0
    epochs = 10
    for epoch in range(epochs):
        counter = train(epoch)
        c += counter
        test()
    # draw_loss(c, epochs)
    print("acc_list:", acc_list)
    print("loss_list:", loss_list)
    torch.save(model, 'model.pth')  # 保存模型

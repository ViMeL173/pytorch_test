# coding=gbk

from DAMG_Normalize import Normalize_DAMG
from Module import Net
from DAGM2007 import DAGM2007
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# ����batch size�����ѵ������ֵ�;�ֵ����transform
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
model.to(device)  # GPUѵ��

criterion = torch.nn.CrossEntropyLoss()  # ��������ʧ log_softmax+NLL_loss
LR = 0.001  # ��ʼѧϰ��
# �Ż�����ʼ�� L2���򻯣�����ϵ��=5*10^-5����ѧϰ�ʡ�����ϵ��=0.9
optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=5e-5, momentum=0.9)  # weight_decay=5e-5,momentum=0.9

loss_list = []
acc_list = []


def Learning_rate_attenuation(lr, idx):  # ѧϰ��˥��������˥��ϵ��Ϊ0.012
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
        if idx % 30 == 29:  # ÿ30�����һ��loss
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
            _, predicted = torch.max(outputs.data, dim=1)  # ȡһ�������ֵ���±��
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # ������ȷ�ĸ���
    print('��ȷ��: %d %%' % (100 * correct / total))
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
    torch.save(model, 'model.pth')  # ����ģ��

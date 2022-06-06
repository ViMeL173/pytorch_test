# coding=gbk

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from ResNet import ResNet18
import os

# �����Ƿ�ʹ��GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ��������,ʹ�������ܹ��ֶ����������в����������÷���ú�Linux�����в��
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model_CIFAR/', help='fo lder to output images and model checkpoints') #����������·��
args = parser.parse_args()

# ����������
EPOCH = 100  # �������ݼ�����
pre_epoch = 0  # �����Ѿ��������ݼ��Ĵ���
BATCH_SIZE = 128  # ������ߴ�(batch_size)
LR = 0.01  # ѧϰ��

# ׼�����ݼ���Ԥ����
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # ���������0���ڰ�ͼ������ü���32*32
    transforms.RandomHorizontalFlip(),  # ͼ��һ��ĸ��ʷ�ת��һ��ĸ��ʲ���ת
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,Bÿ��Ĺ�һ���õ��ľ�ֵ�ͷ���
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=False, transform=transform_train)  # ѵ�����ݼ�
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=2)  # ����һ����batch������ѵ�������batch��ʱ��˳�����ȡ

testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# Cifar-10�ı�ǩ
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ģ�Ͷ���-ResNet
net = ResNet18().to(device)

# ������ʧ�������Ż���ʽ
criterion = nn.CrossEntropyLoss()  # ��ʧ����Ϊ�����أ������ڶ��������
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                      weight_decay=5e-4)  # �Ż���ʽΪmini-batch momentum-SGD��������L2���򻯣�Ȩ��˥����

# ѵ��
if __name__ == "__main__":
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    best_acc = 85  # 2 ��ʼ��best test accuracy
    print("Start Training, ResNet18!")  # ����������ݼ��Ĵ���
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:  # ��׼ȷ�ʺ���ʧ����txt�ļ���
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # ׼������
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # ÿѵ��1��batch��ӡһ��loss��׼ȷ��
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    # д���ļ���
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # ÿѵ����һ��epoch����һ��׼ȷ��
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # ȡ�÷���ߵ��Ǹ��� (outputs.data��������)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('���Է���׼ȷ��Ϊ��%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total

                    # ��ÿ�β��Խ��ʵʱд��acc.txt�ļ���
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # ��¼��Ѳ��Է���׼ȷ�ʲ�д��best_acc.txt�ļ���
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)

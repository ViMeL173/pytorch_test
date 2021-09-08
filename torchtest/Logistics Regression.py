import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0]])

n = 1000


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(size_average=False)  # BCE(二值交叉熵)损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 优化器（用于更新w）

for epoch in range(n):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = np.linspace(0, 10, num=200)
x_t = torch.tensor(x).view((200, 1))
x_t = x_t.to(torch.float32)
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('hours')
plt.ylabel('pro of pass')
plt.grid()
plt.show()

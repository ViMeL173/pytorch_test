import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

a = 0.01
n = 100
w1 = torch.tensor([1.0])
w1.requires_grad = True
w2 = torch.tensor([1.0])
w2.requires_grad = True
b = torch.tensor([1.0])
b.requires_grad = True


def forward(x):
    return w1 * x * x + w2 * x + b


def loss(x, y):
    y_hat = forward(x)
    return (y_hat - y) ** 2


print('Predict (before training)', 4, forward(4).item())

for epoch in range(n):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        w1.data = w1.data - a * w1.grad.data
        w1.grad.data.zero_()
        w2.data = w2.data - a * w2.grad.data
        w2.grad.data.zero_()
        b.data = b.data - a * b.grad.data
        b.grad.data.zero_()

    print("progress:", epoch, l.item())

print("predict (after training)", 4, forward(4).item())
print("predict (after training)", 1, forward(1).item())

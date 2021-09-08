x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0
a = 0.01


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


def cost(xs, ys):
    costs = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        costs += (y_pred - y) ** 2
    return costs / len(xs)


def gradient_rand(x, y):
    return 2 * x * (x * w - y)


def gradient(xs, ys):   # loss对w求导
    grads = 0
    for x, y in zip(xs, ys):
        grads += 2 * x * (x * w - y)
    return grads / len(xs)


print('Predict (before training)', 4, forward(4))

for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= a * grad_val

    print("progress:", epoch, "w=", w, "loss=", cost_val)

print('Predict (after training)', 4, forward(4))

import numpy as np
import torch

xy_test = np.loadtxt('diabetes_test.csv', delimiter=',', dtype=np.float32)
x_test = torch.from_numpy(xy_test[:, :-1])
y_test = torch.from_numpy(xy_test[:, [-1]])

print(y_test[1].item())

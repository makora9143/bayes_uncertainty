import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

from mcdp.layers.dropout import MCDropout
from mcdp.layers.linear import GaussianLinear


def f(x, train=False):
    if train:
        eps = np.random.normal(0, 0.3, x.shape)
    else:
        eps = 0
    # return x * (np.sin(-2*(x+2) + eps) - x *  4 * np.sin(3*(x+ eps)) +  eps - np.cos(3 * x))
    return x * (np.sin(-2*(x+2)) - x *  4 * np.sin(3*(x)) +  eps - np.cos(3 * x))

class PointDataset(data.Dataset):
    def __init__(self):
        super(PointDataset, self).__init__()

        train_x = np.random.uniform(-2, -0.5, (2000, 1)).astype('float32')
        train_x3 = np.random.uniform(0.3, 0.7, (1000, 1)).astype('float32')
        train_x2 = np.random.uniform(1.5, 2, (2000, 1)).astype('float32')
        train_x = np.concatenate([train_x, train_x3, train_x2], 0)
        train_y = f(train_x, True).astype('float32')

        self.train_x, self.train_y = torch.from_numpy(train_x), torch.from_numpy(train_y)

    def __len__(self):
        return self.train_x.size(0)

    def __getitem__(self, index):
        return self.train_x[index].view(1), self.train_y[index].view(1)


class Model(nn.Module):
    def __init__(self, l, k, L, activation, p=0.5):
        super(Model, self).__init__()

        self.net = nn.Sequential(
            # MCDropout(0.5),
            GaussianLinear(1, k, l),
            activation(),
            MCDropout(p),
        )
        for i in range(L):
            self.net.add_module('({})'.format(i+1),
                                nn.Sequential(
                                    GaussianLinear(k, k, l),
                                    activation(),
                                    MCDropout(p),
                                ))
        self.net.add_module('{}'.format(L+1), GaussianLinear(k, 1, l))

    def forward(self, x):
        return self.net(x)


true_x = np.linspace(-2, 2, 5000)
true_y = f(true_x)

train_dataset = PointDataset()
loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)

test_x = torch.linspace(-4, 4, 400).view(-1, 1)

model = Model(5, 1024, 4, nn.ReLU)
print(model)
optimizer = optim.Adam(model.parameters(), weight_decay=25 * 0.5 / 2 / 5000 / 10, amsgrad=True)
creterion = nn.MSELoss()

epochs = 10
batch_size = 32

for epoch in range(epochs):
    losses = 0
    for x, y in loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = creterion(pred, y)
        loss.backward()
        optimizer.step()
        losses += loss.item()
    print(epoch, losses / len(loader))



model.eval()
tmp = []
for i in range(10):
    tmp.append(model(test_x))
total = torch.cat(tmp, 1)
test_y = total.mean(1)

var = torch.pow(total, 2).mean(1) - torch.pow(test_y, 2)
std = torch.sqrt(var)

# plt.plot(train_dataset.train_x.detach().numpy(), train_dataset.train_y.detach().numpy(), 'gx', linewidth=0.1)
# plt.plot(true_x, true_y, 'b-')
plt.plot(test_x.detach().numpy(), test_y.detach().numpy(),'b')
plt.fill_between(test_x.detach().numpy().reshape(-1), (test_y - 2 * std).detach().numpy(), (test_y + 2 * std).detach().numpy(), alpha=0.2)
plt.show()




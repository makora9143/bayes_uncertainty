import torch
import torch.nn as nn
from mcdp.layers.dropout import MCDropout
from mcdp.layers.linear import PriorLinear

import matplotlib.pyplot as plt


params = {
    "ReLU": [[1, 10], [32, 512, 4096]],
    "Tanh": [[1, 10], [32, 512, 4096]],
    "Sigmoid": [[0.1, 1], [32, 512, 4096]]
}

activations = {
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid
}


def plot(l, k, func, idx):
    activation = activations[func]

    drop = nn.Sequential(
        PriorLinear(1, k, l),
        activation(),
        MCDropout(), 
        PriorLinear(k, k, l),
        activation(),
        MCDropout(), 
        PriorLinear(k, k, l),
        activation(),
        MCDropout(), 
        PriorLinear(k, 1, l)
    )

    x = torch.linspace(-2, 2, 100).view(-1, 1)

    drop.eval()
    tmp = []
    for i in range(20):
        y = drop(x)
        tmp.append(y)
        plt.subplot(6, 3, idx)
        plt.plot(x.detach().numpy(), y.detach().numpy(), 'c', linewidth=0.3)
        plt.title('{}, l:{}, k:{}'.format(func, l, k))
    y_mean = torch.mean(torch.cat(tmp, 1), 1).view(-1, 1)
    plt.plot(x.detach().numpy(), y_mean.detach().numpy(), 'b', linewidth=1.0)

plt.subplots_adjust(wspace=0.4, hspace=0.6)

for i, (activation, params) in enumerate(params.items(), 1):
    ls, ks = params
    j = 0
    for k in ks:
        for l in ls:
            plot(l, k, activation, i + (3 * j))

            j += 1




plt.show()

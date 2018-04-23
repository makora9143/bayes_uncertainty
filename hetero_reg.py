import itertools
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.animation as anime

from tqdm import tqdm

from mcdp.models import HeteroMCDropout
from mcdp.dataset import PointDataset
from mcdp.loss import HeteroGaussianNLLLoss


def train_epoch(model, creterion, dataloader, optimizer):
    model.predict()
    losses = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        x, y = Variable(x), Variable(y)
        mean, sigma2 = model(x)
        # loss = creterion(mean, y)
        loss = creterion(mean, y, 1./sigma2)
        loss.backward()
        optimizer.step()
        losses += loss.data.item()
    return losses / len(dataloader) / dataloader.batch_size

def test(model, dataloader):
    model.mc()
    with torch.autograd.no_grad():
        x, y = iter(dataloader).next()
    # x, y = Variable(x), Variable(y)
    mean, var = model(x)
    return mean, torch.sqrt(var)

def xsin(x):
    return x * torch.sin(x)

def main(args):
    func = xsin

    train_dataset = PointDataset(args.N, function=func)
    trainloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = PointDataset(low=-12, high=12, function=func)
    testloader = data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    model = HeteroMCDropout(args.drop_p, args.units, args.sampling)
    print(model)

    l2_decay = args.l2 * (1 - args.drop_p) / (2 * args.N)
    print('L2_decay =', l2_decay)

    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=l2_decay)
    # optimizer = optim.Adam(model.parameters(), weight_decay=l2_decay)

    # creterion = nn.MSELoss()
    creterion = HeteroGaussianNLLLoss()

    artists = []
    fig, ax = plt.subplots()
    ax.set_ylim(-10, 10)
    b = ax.plot(train_dataset.xs.numpy(), train_dataset.ys.numpy(), 'o', color='black')
    
    pbar = tqdm(range(1, args.epochs + 1))
    colors = ['steelblue', 'deepskyblue', 'lightskyblue', 'aliceblue']

    for epoch in pbar:
        train_loss = train_epoch(model, creterion, trainloader, optimizer)
        mean, std = test(model, testloader)

        c = ax.plot(test_dataset.xs.numpy(), mean.data.numpy(), 'b', linewidth=1.5)
        for i in range(0, 3):
            c += ax.plot(test_dataset.xs.numpy(),
                         (mean - 2 * std * (i+1)/4).data.numpy(),
                         '--', color=colors[i], linewidth=0.5)
            c += ax.plot(test_dataset.xs.numpy(),
                         (mean + 2 * std * (i+1)/4).data.numpy(),
                         '--', color=colors[i], linewidth=0.5)

        artists.append(b + c)
        pbar.set_description('Epoch {}'.format(epoch))
        pbar.set_postfix(train=train_loss)


    animation = anime.ArtistAnimation(fig, artists, interval=100, repeat_delay=1500)
    # animation.save('certainity_regression.gif', writer='imagemagick', fps=100)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ArgParser')

    parser.add_argument('--sampling', type=int, default=100,
                        help='Number of sampling')
    parser.add_argument('--N', type=int, default=20,
                        help='Number of data')
    parser.add_argument('--units', type=int, default=20,
                        help='Number of hidden units')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Epoch')
    parser.add_argument('--lam', type=float, default=0.00001,
                        help='weight for L2 norm')
    parser.add_argument('--drop_p', type=float, default=0.05,
                        help='dropout probability')
    parser.add_argument('--l2', type=float, default=0.1,
                        help='data length frequency')

    args = parser.parse_args()
    main(args)

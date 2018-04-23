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

from mcdp.models import MCDropout
from mcdp.dataset import PointDataset


def train_epoch(model, creterion, dataloader, optimizer):
    model.train()
    model.not_estimate()
    losses = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        x, y = Variable(x), Variable(y)
        mean = model(x)
        loss = creterion(mean, y)
        loss.backward()
        optimizer.step()
        losses += loss.data.item()
    return losses / len(dataloader)

def test(model, creterion, dataloader):
    model.estimate()
    model.train()
    x, y = iter(dataloader).next()
    x, y = Variable(x), Variable(y)
    predict, var, result = model(x)
    loss = creterion(predict, y)
    return predict, var, result, loss.data.item()


def xsin(x):
    return x * torch.sin(x)

def main(args):

    train_dataset = PointDataset(args.N, low=-5, high=5, function=xsin)
    trainloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = PointDataset(low=-7, high=7, function=xsin)
    testloader = data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    model = MCDropout(args.drop_p, args.units, args.sampling)
    print(model)

    optimizer = optim.Adam(model.parameters(), weight_decay=args.lam)

    creterion = nn.MSELoss()

    artists = []
    fig, ax = plt.subplots()
    a = ax.plot(test_dataset.xs.numpy(), test_dataset.ys.numpy(), 'r')
    b = ax.plot(train_dataset.xs.numpy(), train_dataset.ys.numpy(), 'bo')

    
    pbar = tqdm(range(1, args.epochs + 1))

    for epoch in pbar:
        train_loss = train_epoch(model, creterion, trainloader, optimizer)
        mean, var, result, test_loss = test(model, creterion, testloader)
        std = torch.sqrt(var) + 2 * args.N * args.lam / (1 - args.drop_p) / 0.005
        c = ax.plot(test_dataset.xs.numpy(), mean.data.numpy(), 'b')
        d = ax.plot(test_dataset.xs.numpy(), (mean - std).data.numpy(), 'c--')
        e = ax.plot(test_dataset.xs.numpy(), (mean + std).data.numpy(), 'c--')
        f = ax.plot(test_dataset.xs.numpy(), result[0].data.numpy(), 'gray')
        # g = ax.fill_between(test_dataset.xs.numpy(),(mean - std).data.numpy().reshape(-1) , (mean + std).data.numpy().reshape(-1))

        artists.append(a + b + c + d + e + f )
        pbar.set_description('Epoch{}'.format(epoch))
        pbar.set_postfix(train=train_loss, test=test_loss)


    animation = anime.ArtistAnimation(fig, artists, interval=100, repeat_delay=1500)
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

    args = parser.parse_args()
    main(args)

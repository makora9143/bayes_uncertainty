import pickle
import math
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.autograd import Variable

from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from mcdp.models import MCLeNet



def train(epoch, model, dataloader, creterion, optimizer, use_cuda=False):
    model.predict()

    pbar = tqdm(dataloader)
    losses = 0
    pbar.set_description('Epoch {}'.format(epoch))
    for x, y in pbar:
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)
        optimizer.zero_grad()
        predict = model(x)
        loss = F.nll_loss(predict, y)
        loss.backward()
        optimizer.step()
        losses += loss.data.item()
        pbar.set_postfix(loss=math.exp(loss.data.item()))
    return losses / len(dataloader.dataset)

def test(model, creterion, dataloader, use_cuda=False):
    # model.mc()

    test_loss = 0
    correct = 0
    pbar = tqdm(dataloader)
    for x, y in pbar:
        with torch.no_grad():
            x, y = Variable(x), Variable(y)
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        mean = model(x)
        loss = creterion(mean, y)
        test_loss += math.exp(loss.data.item())
        pbar.set_postfix(loss=math.exp(loss.data.item()))
        pred = mean.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(y.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(dataloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))
       
def scatter(output, labels=[1, 5, 7]):
    sample = output.shape[0]
    num_img = output.shape[1]
    x = np.array([list(range(num_img))] * sample).T.reshape(-1)
    # plt.figure(figsize=(12, 9))
    y = output[:, :, 1].T.reshape(-1,)
    plt.scatter(x, y, s=1500, marker='_', alpha=0.3, label='1')
    y = output[:, :, 5].T.reshape(-1,)
    plt.scatter(x, y, s=1000, marker='_', c='orange', alpha=0.3, label='5')
    y = output[:, :, 7].T.reshape(-1,)
    plt.scatter(x, y, s=1000, marker='_', c='green', alpha=0.3, label='7')
    plt.legend(loc='center right')
    plt.savefig('classification_uncertainity.png')

def main(args):

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('~/data/mnist/', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('~/data/mnist/', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    with open('./test_img.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    model = MCLeNet()
    if args.cuda:
        print('use cuda')
        model.cuda()

    l2_decay = args.l2 * (1 - args.drop_p) / (2 * args.N)
    print('L2_decay =', l2_decay)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=l2_decay)

    creterion = nn.NLLLoss()

    for epoch in range(1, args.epochs+1):
        losses = train(epoch, model, train_loader, creterion, optimizer, args.cuda)
        test(model, creterion, test_loader, args.cuda)

    toy_data = Variable(data_dict['tensor']).cuda() if args.cuda else Variable(data_dict['tensor'])
    model.mc()
    uncertainity = torch.exp(model(toy_data))

    scatter(uncertainity.data.cpu().numpy())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ArgParser')
    parser.add_argument('--sampling', type=int, default=10,
                        help='Number of sampling')
    parser.add_argument('--N', type=int, default=20,
                        help='Number of data')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Epoch')
    parser.add_argument('--lam', type=float, default=0.00001,
                        help='weight for L2 norm')
    parser.add_argument('--drop_p', type=float, default=0.05,
                        help='dropout probability')
    parser.add_argument('--l2', type=float, default=0.1,
                        help='data length frequency')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    args.cuda =  args.use_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    main(args)

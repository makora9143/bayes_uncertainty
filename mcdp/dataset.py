import torch
import torch.utils.data as data

class PointDataset(data.Dataset):
    def __init__(self, N=None, low=-1, high=7, function=torch.sin, noise=False):
        super(PointDataset, self).__init__()

        self.N = N
        self.function = function

        self.low = low
        self.high = high
        self.noise = noise

        # self.xs, self.ys = self.sample()
        self.xs, self.ys = self.reproduce_sample()

    def __len__(self):
        return self.xs.size(0)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        x, y = self.xs[index].view(1), self.ys[index].view(1)

        return x, y

    def sample(self):
        if self.N is None:
            xs = torch.arange(self.low, self.high, (self.high - self.low) / 140)
        else:
            xs = torch.rand(self.N).uniform_(self.low, self.high)
        ys = self.function(xs)

        if self.noise:
            ys += torch.rand(ys.size()).normal_(0, 0.01)

        return xs, ys

    def reproduce_sample(self):
        if self.N is None:
            xs = torch.arange(self.low, self.high, (self.high - self.low) / 200)
            ys = self.function(xs)
        else:
            xs = torch.FloatTensor([1. * i / (self.N - 5) * 10 - 5 for i in range(self.N - 4)])
            ys = self.function(xs)

            xs = torch.FloatTensor(xs.tolist() + [7, 8.5, 10, 11.5])
            ys = torch.FloatTensor(ys.tolist() + [-7, 7, -7, 7])
        return xs, ys


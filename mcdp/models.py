import itertools

import torch
import torch.nn as nn

from .layers import HeteroLinear, GaussianLinear, Flatten

class MCDropout(nn.Module):
    def __init__(self, drop_p=0.1, hidden=20, sampling=100):
        super(MCDropout, self).__init__()
        self.net = nn.Sequential(

            # nn.Dropout(0.05),
            nn.Linear(1, hidden),
            nn.ReLU(),

            nn.Dropout(drop_p),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),

            nn.Dropout(drop_p),
            nn.Linear(hidden, 1)
        )
        
        self.sampling = sampling
        self.mc_flg = False

    def forward(self, x):
        if self.mc_flg:
            result = self.mc_forward(x)
            mean = torch.mean(result, 0)
            var = torch.mean(torch.pow(result, 2), 0) - torch.pow(mean, 2)
            return mean, var, result
        else:
            return self.net(x)

    def mc(self):
        self.mc_flg = True
        return self

    def predict(self):
        self.mc_flg = False
        return self

    def mc_forward(self, x):
        xs = itertools.repeat(x, self.sampling)
        result = torch.stack(list(map(self.net, xs)))
        return result



class HeteroMCDropout(MCDropout):
    def __init__(self, drop_p=0.1, hidden=20, sampling=100):
        super(HeteroMCDropout, self).__init__(drop_p=0.1, hidden=20, sampling=100)
        self.net = nn.Sequential(
            # nn.Dropout(0.05),
            nn.Linear(1, hidden),
            nn.ReLU(),

            nn.Dropout(drop_p),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),

            nn.Dropout(drop_p),
            # nn.Linear(hidden, 2)
            HeteroLinear(hidden, 2),
        )

    def forward(self, x):
        if self.mc_flg:
            result = self.mc_forward(x)
            mc_mean, mc_sigma2 = self.separate(result)

            mean = torch.mean(mc_mean, 0)
            var = self.variance(mc_mean, mc_sigma2)

            return mean, var
        else:
            result = self.net(x)
            mean, sigma2 = self.separate(result)
            return mean, sigma2

    def separate(self, x):
        mean = x.index_select(-1, torch.LongTensor([0]))
        sigma2 = torch.exp(x.index_select(-1, torch.LongTensor([1])))
        return mean, sigma2


    def variance(self, mc_mean, mc_sigma2):
        exp_mean = torch.mean(mc_mean, 0)
        exp_sigma2 = torch.mean(mc_sigma2, 0)
        var = torch.mean(torch.pow(mc_mean, 2), 0) - torch.pow(exp_mean, 2) + exp_sigma2
        return var


class MCLeNet(MCDropout):

    def __init__(self, drop_p=0.1, hidden=20, sampling=100):
        super(MCLeNet, self).__init__(drop_p=0.1, hidden=20, sampling=100)

        self.net = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(20, 50, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            Flatten(),
            
            nn.Linear(50* 4 * 4, 500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500, 10),
            nn.LogSoftmax(-1)
        )

    def forward(self, x):
        if self.mc_flg:
            result = self.mc_forward(x)
        else:
            result = self.net(x)
        return result


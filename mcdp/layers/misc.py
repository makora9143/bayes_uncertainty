import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self,x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)



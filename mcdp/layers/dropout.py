import torch
import torch.nn as nn
from torch.autograd.function import InplaceFunction
from torch.nn.modules.dropout import _DropoutNd


class MCDropoutFunction(InplaceFunction):
    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @staticmethod
    def symbolic(g, input, p=0.5, train=False, inplace=False):
        # See Note [Export inplace]
        r, _ = g.op("Dropout", input, ratio_f=p, is_test_i=not train, outputs=2)
        return r

    @classmethod
    def forward(cls, ctx, input, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p == 0 or not ctx.train:
            noise = cls._make_noise(output.narrow(0, 0, 1))
            noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
            noise.expand_as(output)
            output.mul_(noise)
            return output

        ctx.noise = cls._make_noise(input)
        if ctx.p == 1:
            ctx.noise.fill_(0)
        else:
            ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
        ctx.noise = ctx.noise.expand_as(input)
        output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output * ctx.noise, None, None, None
        else:
            return grad_output, None, None, None


def mc_dropout(input, p=0.5, training=False, inplace=False):
    return MCDropoutFunction.apply(input, p, training, inplace )


class MCDropout(_DropoutNd):
    def __init__(self, p=0.5, inplace=False, sampling=100):
        super(MCDropout, self).__init__(p=0.5, inplace=False)
    
    def forward(self, input):
        output = mc_dropout(input, self.p, self.training, self.inplace)
        return output

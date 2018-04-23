import math
import torch

from torch.nn.modules.loss import _Loss


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

def gnll_loss(input, target, prec, size_average=True, reduce=True):
    return _pointwise_loss(lambda a, b, c: 0.5 * (math.log(2 * math.pi) - torch.log(c) + c * (a - b) ** 2), torch._C._nn.mse_loss,
                           input, target, prec, size_average, reduce)


def _pointwise_loss(lambd, lambd_optimized, input, target, prec, size_average=True, reduce=True):
    if target.requires_grad:
        d = lambd(input, target, prec)
        if not reduce:
            return d
        return torch.mean(d) if size_average else torch.sum(d)
    else:
        return lambd_optimized(input, target, size_average, reduce)


class HeteroGaussianNLLLoss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(HeteroGaussianNLLLoss, self).__init__(size_average, reduce)

    def forward(self, input, target, prec):
        _assert_no_grad(target)
        return gnll_loss(input, target, prec, size_average=self.size_average, reduce=self.reduce)


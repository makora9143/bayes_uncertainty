import torch
from torch.autograd import Variable
from torch.autograd import gradcheck
from mcdp.heterolayer import HeteroLinearFunction, LinearFunction

input = (Variable(torch.randn(60,20).double(), requires_grad=True), Variable(torch.randn(2,20).double(), requires_grad=True),Variable(torch.randn(2,).double(), requires_grad=True))
# test = gradcheck(LinearFunction.apply, input, eps=1e-6, atol=1e-4)
test = gradcheck(HeteroLinearFunction.apply, input, eps=1e-6, atol=1e-4)

print(test)

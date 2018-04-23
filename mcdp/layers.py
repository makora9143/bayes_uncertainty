import math
import torch
import torch.nn as nn
from torch.autograd import Function


class HeteroLinearFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, output = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        grad_output_mean = grad_output.index_select(1, torch.LongTensor([0]))
        prec = torch.exp(-output.index_select(1, torch.LongTensor([1])))
        weight_mean = weight.index_select(0, torch.LongTensor([0]))
        weight_sigma2 = weight.index_select(0, torch.LongTensor([1]))


        if ctx.needs_input_grad[0]:
            grad_input = (prec * grad_output_mean).mm(weight_mean)
            grad_input -= 0.5 * (prec * grad_output_mean ** 2 - 1).mm(weight_sigma2)
        if ctx.needs_input_grad[1]:
            grad_weight_mean = (prec * grad_output_mean).t().mm(input)
            grad_weight_sigma2 = - 0.5 * (prec * grad_output_mean ** 2).t().mm(input)
            grad_weight = torch.cat([grad_weight_mean, grad_weight_sigma2], 0)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias_mean = (prec * grad_output_mean).sum(0)
            grad_bias_sigma2 = (- 0.5 * (prec * grad_output_mean ** 2 - 1)).sum(0)
            grad_bias = torch.cat([grad_bias_mean, grad_bias_sigma2])

        return grad_input, grad_weight, grad_bias

class HeteroLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(HeteroLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        self.weight.data.normal_(0., 1./input_features)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return HeteroLinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )


class GaussianLinear(nn.Linear):
    def __init__(self, in_features, out_features, scale=None, bias=True):
        super(GaussianLinear, self).__init__(in_features, out_features, bias=True)
        self.scale = scale

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.weight.data.normal_(0., math.sqrt(1./in_features))
        if bias is not None:
            self.bias.data.zero_()


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self,x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)



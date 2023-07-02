import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from functools import partial
from pdb import set_trace as st

def identity(x):
    return x

# Basic SIREN fully connected layer
class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, std_init=1, freq_init=False, is_first=False, act=False):
        super().__init__()
        if is_first:
            self.weight = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-1 / in_dim, 1 / in_dim))
        elif freq_init:
            self.weight = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-np.sqrt(6 / in_dim) / 25, np.sqrt(6 / in_dim) / 25))
        else:
            self.weight = nn.Parameter(0.25 * nn.init.kaiming_normal_(torch.randn(out_dim, in_dim), a=0.2, mode='fan_in', nonlinearity='leaky_relu'))

        self.bias = nn.Parameter(nn.init.uniform_(torch.empty(out_dim), a=-np.sqrt(1/in_dim), b=np.sqrt(1/in_dim)))

        self.bias_init = bias_init
        self.std_init = std_init
        self.activation = torch.sin if act else identity

    def forward(self, input):
        out = self.std_init * F.linear(input, self.weight, bias=self.bias) + self.bias_init
        out = self.activation(out)

        return out

# Siren Generator Model
class OccupancyMap(nn.Module):
    def __init__(self, args):
        super(OccupancyMap, self).__init__()
        self.D = args.D
        self.W = args.W
        self.inc = args.inc
        self.outc = args.outc
        self.activation = torch.sin

        self.pts_linears = nn.ModuleList(
            [LinearLayer(self.inc, self.W, is_first=True, act=True)] +
            [LinearLayer(self.W, self.W, freq_init=True, act=True) for i in range(self.D-1)])

        self.sigma_linear = LinearLayer(self.W, self.outc, freq_init=True, act=False)

    def forward(self, x):
        out = x
        for i in range(len(self.pts_linears)):
            out = self.pts_linears[i](out)

        out = self.sigma_linear(out)

        return out

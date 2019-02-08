"""Module interfaces for the custom operators."""

import math
import torch as th
from torch import nn

from .. import functions as F

class KernelLookup(nn.Module):
    def __init__(self, c_in, ksize, nkernels):
        super(KernelLookup, self).__init__()
        self.c_in = c_in
        self.ksize = ksize
        self.nkernels = nkernels
        self.weights = nn.Parameter(th.ones(nkernels, c_in, ksize, ksize))

        self.init_weights()

    def init_weights(self):
        std = 1.0 / math.sqrt(self.ksize*self.ksize*self.c_in)
        th.nn.init.normal_(self.weights, std=std)

        # th.nn.init.uniform_(self.weights, 0, 1)

        # nrm = self.weights.detach().abs().sum(-1, keepdim=True).sum(-2, keepdim=True)
        # self.weights.data /= nrm

    def __repr__(self):
        s = "KernelLookup(c_in={}, ksize={}, nkernels={})".format(
            self.c_in, self.ksize, self.nkernels)
        return s

    def forward(self, data, kernel_idx):
        n, c, h, w = self.weights.shape
        weights = self.weights
        # nrm = self.weights.detach().abs().sum(-1, keepdim=True).sum(-2, keepdim=True)
        # weights = weights / (nrm+1e-8)
        weights = weights.view(n, c, h*w)
        weights = th.nn.functional.softmax(weights, dim=-1).view(n, c, h, w)
        return F.KernelLookup.apply(data, kernel_idx, weights)

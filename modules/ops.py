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
        # self.weights.data[..., 1, 1] = 1.0
        # th.nn.init.uniform_(self.weights, 0, 2)
        th.nn.init.normal_(self.weights, std=std)

        # self.weights.data.fill_(0.0)
        # self.weights.data.fill_(0.5)
        # self.weights.data[:, :, (self.ksize-1)//2, (self.ksize-1)//2] = 1.0

        # nrm = self.weights.detach().abs().sum(-1, keepdim=True).sum(-2, keepdim=True)
        # self.weights.data /= nrm
        # th.nn.init.constant_(self.weights[0, 0, 1, 1], 1.0)
        # th.nn.init.constant_(self.weights[1, 0, 0, 1], 1.0)
        # th.nn.init.constant_(self.weights[2, 0, 1, 0], 1.0)
        # th.nn.init.constant_(self.weights[3, 0, 1, 2], 1.0)

    def __repr__(self):
        s = "KernelLookup(c_in={}, ksize={}, nkernels={})".format(
            self.c_in, self.ksize, self.nkernels)
        return s

    def forward(self, data, kernel_idx):
        n, c, h, w = self.weights.shape
        weights = self.weights
        # weights = weights.view(n, c, h*w)
        # weights = th.nn.functional.softmax(weights, dim=-1).view(n, c, h, w)
        return F.KernelLookup.apply(data, kernel_idx, weights)

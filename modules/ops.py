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
        self.weights = nn.Parameter(th.zeros(nkernels, c_in, ksize, ksize))

        self.init_weights()

    def init_weights(self):
        std = 1.0 / math.sqrt(self.ksize*self.ksize*self.c_in)
        th.nn.init.normal_(self.weights, std=std)

    def __repr__(self):
        s = "KernelLookup(c_in={}, ksize={}, nkernels={})".format(
            self.c_in, self.ksize, self.nkernels)
        return s

    def forward(self, data, kernel_idx):
        return F.KernelLookup.apply(data, kernel_idx, self.weights)

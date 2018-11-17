from collections import OrderedDict

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class _ConvModule(nn.Module):
  def __init__(self, ninputs, ksize, noutputs, normalize=False, 
               normalization_type="batch", stride=1, padding=0,
               activation="relu", weight_norm=True):
    super(_ConvModule, self).__init__()
    if activation == "relu":
      act_fn = nn.ReLU
    elif activation == "leaky_relu":
      act_fn = nn.LeakyReLU
    elif activation == "tanh":
      act_fn = nn.Tanh
    elif activation == "elu":
      act_fn = nn.ELU
    else:
      raise NotImplemented

    if normalize:
      conv = nn.Conv2d(ninputs, noutputs, ksize, stride=stride, padding=padding, bias=False)
      if normalization_type == "batch":
        nrm = nn.BatchNorm2d(noutputs)
      elif normalization_type == "instance":
        nrm = nn.InstanceNorm2D(noutputs)
      else:
        raise ValueError("Unkown normalization type {}".format(normalization_type))
      nrm.bias.data.zero_()
      nrm.weight.data.fill_(1.0)
      self.layer = nn.Sequential(conv, nrm, act_fn())
    else:
      conv = nn.Conv2d(ninputs, noutputs, ksize, stride=stride, padding=padding)
      if weight_norm:
        conv = nn.utils.weight_norm(conv)  # TODO
      conv.bias.data.zero_()
      self.layer = nn.Sequential(conv, act_fn())

    if activation == "elu":
      nn.init.xavier_uniform_(conv.weight.data, nn.init.calculate_gain("relu"))
    else:
      nn.init.xavier_uniform_(conv.weight.data, nn.init.calculate_gain(activation))

  def forward(self, x):
    out = self.layer(x)
    return out

class ConvChain(nn.Module):
  """Linear chain of convolution layers with no spatial downsampling.

  Args:
    n_in(int): number of input channels.
    n_out(int): number of output channels.
    ksize(int): size of the convolution kernel (square).
    width(int): number of features channels in the intermediate layers.
    activation(): nonlinear activation function between convolutions.
    pad(bool): if True, zero pad the convolutions to maintain a constant size.
    norm_layer(): normalization to apply between the convolution modules.
    out_activation(): activation function applied to the output, defaults to linear (none).
  """
  def __init__(self, n_in, n_out, ksize=3, width=64, depth=3,
               activation=nn.ReLU(inplace=True), pad=True, norm_layer=None,
               out_activation=None):
    super(ConvChain, self).__init__()

    assert isinstance(n_in, int) and n_in > 0, "Input channels should be a positive integer"
    assert isinstance(n_out, int) and n_out > 0, "Output channels should be a positive integer"
    assert isinstance(ksize, int) and ksize > 0, "Kernel size should be a positive integer"
    assert isinstance(depth, int) and depth > 0, "Depth should be a positive integer"

    if pad:
      padding = ksize//2
    else:
      padding = 0

    layers = []
    for d in range(depth-1):
      if d == 0:
        _in = n_in
      else:
        _in = width
      layers.append(_ConvModule(_in, ksize, width, normalize=normalize,
                                normalization_type="batch", padding=padding,
                                stride=1, activation=activation))

    # Last layer
    if depth > 1:
      _in = width
    else:
      _in = n_in

    conv = nn.Conv2d(_in, n_out, ksize, bias=True, padding=padding)
    conv.bias.data.zero_()
    if output_type == "elu" or output_type == "softplus":
      nn.init.xavier_uniform_(
          conv.weight.data, nn.init.calculate_gain("relu"))
    else:
      nn.init.xavier_uniform_(
          conv.weight.data, nn.init.calculate_gain(output_type))
    layers.append(conv)

    # Rename layers
    for im, m in enumerate(layers):
      if im == len(layers)-1:
        name = "prediction"
      else:
        name = "layer_{}".format(im)
      self.add_module(name, m)

    if out_activation is not None:
      self.add_module("output_activation", out_activation)

  def forward(self, x):
    for m in self.children():
      x = m(x)
    return x

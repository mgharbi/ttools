"""Common used neural networks operations."""

from collections import OrderedDict

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


from torch.autograd import Variable

class _ConvModule(nn.Module):
  """Basic convolution module with conv + norm(optional) + activation(optional).

  Args:
    n_in(int): number of input channels.
    n_out(int): number of output channels.
    ksize(int): size of the convolution kernel (square).
    stride(int): 
    pad(bool): if True, zero pad the convolutions to maintain a constant size.
    activation(str): nonlinear activation function between convolutions.
    norm_layer(str): normalization to apply between the convolution modules.
  """
  def __init__(self, n_in, n_out, ksize, stride=1, pad=True,
               activation="relu", norm_layer=None):
    super(_ConvModule, self).__init__()

    assert isinstance(n_in, int) and n_in > 0, "Input channels should be a positive integer got {}".format(n_in)
    assert isinstance(n_out, int) and n_out > 0, "Output channels should be a positive integer got {}".format(n_out)
    assert isinstance(ksize, int) and ksize > 0, "Kernel size should be a positive integer got {}".format(ksize)

    padding = (ksize - 1) // 2 if pad else 0
    use_bias_in_conv = norm_layer is None

    self.add_module("conv", nn.Conv2d(n_in, n_out, ksize, stride=stride,
                                  padding=padding, bias=use_bias_in_conv))

    if norm_layer is not None:
      self.add_module("norm", _get_norm_layer(norm_layer, n_out))

    if activation is not None:
      self.add_module("activation", _get_activation(activation))

    # Initialize parameters
    _init_fc_or_conv(self.conv, activation)

  def forward(self, x):
    for c in self.children():
      x = c(x)
    return x


class _FCModule(nn.Module):
  """Basic fully connected module with optional dropout.

  Args:
    n_in(int): number of input channels.
    n_out(int): number of output channels.
    activation(str): nonlinear activation function.
    dropout(float): dropout ratio if defined, default to None: no dropout.
  """
  def __init__(self, n_in, n_out, activation="relu", dropout=None):
    super(_FCModule, self).__init__()

    assert isinstance(n_in, int) and n_in > 0, "Input channels should be a positive integer"
    assert isinstance(n_out, int) and n_out > 0, "Output channels should be a positive integer"

    self.add_module("fc", nn.Linear(n_in, n_out))

    if activation is not None:
      self.add_module("activation", _get_activation(activation))

    if dropout is not None:
      self.add_module("dropout", nn.Dropout(dropout, inplace=True))


    # Initialize parameters
    _init_fc_or_conv(self.fc, activation)

  def forward(self, x):
    for c in self.children():
      x = c(x)
    return x


class ConvChain(nn.Module):
  """Linear chain of convolution layers with no spatial downsampling.

  Args:
    n_in(int): number of input channels.
    n_out(int): number of output channels.
    ksize(int): size of the convolution kernel (square).
    width(int): number of features channels in the intermediate layers.
    strides(list of int)
    pad(bool): if True, zero pad the convolutions to maintain a constant size.
    activation(str): nonlinear activation function between convolutions.
    norm_layer(str): normalization to apply between the convolution modules.
    out_activation(): activation function applied to the output, defaults to linear (none).
  """
  def __init__(self, n_in, n_out, ksize=3, width=64, depth=3, strides=None, pad=True,
               activation="relu", norm_layer=None, out_activation=None):
    super(ConvChain, self).__init__()

    assert isinstance(n_in, int) and n_in > 0, "Input channels should be a positive integer"
    assert isinstance(n_out, int) and n_out > 0, "Output channels should be a positive integer"
    assert (isinstance(ksize, int) and ksize > 0) or isinstance(ksize, list), "Kernel size should be a positive integer or a list of integers"
    assert isinstance(depth, int) and depth > 0, "Depth should be a positive integer"
    assert isinstance(width, int) or isinstance(width, list), "Width should be a list or an int"

    _in = [n_in]
    _out = [n_out]

    if strides is None:
      _strides = [1]*depth
    else:
      assert isinstance(strides, list), "strides should be a list"
      assert len(strides) == depth, "strides should have `depth` elements"
      _strides = strides

    if isinstance(width, int):
      _in = _in + [width]*(depth-1)
      _out = [width]*(depth-1) + _out
    elif isinstance(width, list):
      assert depth > 2 and len(width) == depth-1, "Needs at least three layers to specify width with a list."
      _in = _in + width
      _out = width + _out

    if isinstance(ksize, int):
      _ksizes = [ksize]*depth
    elif isinstance(ksize, list):
      assert len(ksize) == depth, "kernel size list should have 'depth' entries"
      _ksizes = ksize

    _activations = [activation]*(depth-1) + [out_activation]
    _norms = [None] + [norm_layer]*(depth-2) + [None]  # dont normalize in/out layers

    # Core processing layers, no norm at the first layer
    for lvl in range(depth):
      self.add_module(
        "conv{}".format(lvl),
        _ConvModule(_in[lvl], _out[lvl], _ksizes[lvl], stride=_strides[lvl], pad=pad,
                    activation=_activations[lvl], norm_layer=_norms[lvl]))

  def forward(self, x):
    for m in self.children():
      x = m(x)
    return x


class FCChain(nn.Module):
  """Linear chain of fully connected layers.

  Args:
    n_in(int): number of input channels.
    n_out(int): number of output channels.
    width(int or list of ints): number of features channels in the intermediate
      layers. Specify an int for a uniform width, or a list for more control.
    depth(int): number of layers.
    activation(): nonlinear activation function between convolutions.
    pad(bool): if True, zero pad the convolutions to maintain a constant size.
    norm_layer(): normalization to apply between the convolution modules.
    out_activation(): activation function applied to the output, defaults to linear (none).
  """
  def __init__(self, n_in, n_out, ksize=3, width=64, depth=3, pad=True,
               activation="relu", norm_layer=None, out_activation=None):
    pass

  def forward(self, x):
    for m in self.children():
      x = m(x)
    return x


def _get_norm_layer(norm_layer, channels):
  valid = ["instance", "batch"]
  assert norm_layer in valid, "norm_layer should be one of {}".format(valid)

  if norm_layer == "instance":
    layer = nn.InstanceNorm2d(channels, affine=True)
  elif norm_layer == "batch":
    layer = nn.BatchNorm2d(channels, affine=True)
  nn.init.constant_(layer.bias, 0.0)
  nn.init.constant_(layer.weight, 1.0)
  return layer


def _get_activation(activation):
  valid = ["relu"]
  assert activation in valid, "activation should be one of {}".format(valid)
  if activation == "relu":
    return nn.ReLU(inplace=True)


def _init_fc_or_conv(fc_conv, activation):
  gain = 1.0
  if activation is not None:
    gain = nn.init.calculate_gain(activation)
  nn.init.xavier_uniform_(fc_conv.weight, gain)
  if fc_conv.bias is not None:
    nn.init.constant_(fc_conv.bias, 0.0)



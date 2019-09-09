"""Common used neural networks operations."""

# TODO(mgharbi): maybe add a norm layer at the output if we specify an activation fn ?


import torch as th
from torch import nn

from ..utils import get_logger


LOG = get_logger(__name__)


class FCModule(nn.Module):
    """Basic fully connected module with optional dropout.

    Args:
      n_in(int): number of input channels.
      n_out(int): number of output channels.
      activation(str): nonlinear activation function.
      dropout(float): dropout ratio if defined, default to None: no dropout.
    """

    def __init__(self, n_in, n_out, activation=None, dropout=None):
        super(FCModule, self).__init__()

        assert isinstance(
            n_in, int) and n_in > 0, "Input channels should be a positive integer"
        assert isinstance(
            n_out, int) and n_out > 0, "Output channels should be a positive integer"

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


class FCChain(nn.Module):
    """Linear chain of fully connected layers.

    Args:
      n_in(int): number of input channels.
      width(int or list of int): number of features channels in the intermediate layers.
      depth(int): number of layers
      activation(str): nonlinear activation function between convolutions.
      dropout(float or list of float): dropout ratio if defined, default to None: no dropout.
    """

    def __init__(self, n_in, width=64, depth=3, activation="relu",
                 dropout=None):
        super(FCChain, self).__init__()

        assert isinstance(
            n_in, int) and n_in > 0, "Input channels should be a positive integer"
        assert isinstance(
            depth, int) and depth > 0, "Depth should be a positive integer"
        assert isinstance(width, int) or isinstance(
            width, list), "Width should be a list or an int"

        _in = [n_in]

        if isinstance(width, int):
            _in = _in + [width]*(depth-1)
            _out = [width]*depth
        elif isinstance(width, list):
            assert len(width) == depth, "Specifying width with a least: should have `depth` entries"
            _in = _in + width[:-1]
            _out = width

        _activations = [activation]*depth

        if dropout is not None:
            assert isinstance(dropout, float) or isinstance(
                dropout, list), "Dropout should be a float or a list of floats"

        if dropout is None or isinstance(dropout, float):
            _dropout = [dropout]*depth
        elif isinstance(dropout, list):
            assert len(dropout) == depth, "When specifying a list of dropout, the list should have 'depth' elements."
            _dropout = dropout

        # Core processing layers, no norm at the first layer
        for lvl in range(depth):
            self.add_module(
                "fc{}".format(lvl),
                FCModule(_in[lvl], _out[lvl], activation=_activations[lvl],
                         dropout=_dropout[lvl]))

    def forward(self, x):
        for m in self.children():
            x = m(x)
        return x


class ConvModule(nn.Module):
    """Basic convolution module with conv + norm(optional) + activation(optional).

    Args:
      n_in(int): number of input channels.
      n_out(int): number of output channels.
      ksize(int): size of the convolution kernel (square).
      stride(int): downsampling factor
      pad(bool): if True, zero pad the convolutions to maintain a constant size.
      activation(str): nonlinear activation function between convolutions.
      norm_layer(str): normalization to apply between the convolution modules.
    """

    def __init__(self, n_in, n_out, ksize=3, stride=1, pad=True,
                 activation=None, norm_layer=None):
        super(ConvModule, self).__init__()

        assert isinstance(
            n_in, int) and n_in > 0, "Input channels should be a positive integer got {}".format(n_in)
        assert isinstance(
            n_out, int) and n_out > 0, "Output channels should be a positive integer got {}".format(n_out)
        assert isinstance(
            ksize, int) and ksize > 0, "Kernel size should be a positive integer got {}".format(ksize)

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


class ConvChain(nn.Module):
    """Linear chain of convolution layers.

    Args:
      n_in(int): number of input channels.
      ksize(int or list of int): size of the convolution kernel (square).
      width(int or list of int): number of features channels in the intermediate layers.
      depth(int): number of layers
      strides(list of int): stride between kernels. If None, defaults to 1 for all.
      pad(bool): if True, zero pad the convolutions to maintain a constant size.
      activation(str): nonlinear activation function between convolutions.
      norm_layer(str): normalization to apply between the convolution modules.
    """

    def __init__(self, n_in, ksize=3, width=64, depth=3, strides=None, pad=True,
                 activation="relu", norm_layer=None):
        super(ConvChain, self).__init__()

        assert isinstance(
            n_in, int) and n_in > 0, "Input channels should be a positive integer"
        assert (isinstance(ksize, int) and ksize > 0) or isinstance(
            ksize, list), "Kernel size should be a positive integer or a list of integers"
        assert isinstance(
            depth, int) and depth > 0, "Depth should be a positive integer"
        assert isinstance(width, int) or isinstance(
            width, list), "Width should be a list or an int"

        _in = [n_in]

        if strides is None:
            _strides = [1]*depth
        else:
            assert isinstance(strides, list), "strides should be a list"
            assert len(strides) == depth, "strides should have `depth` elements"
            _strides = strides

        if isinstance(width, int):
            _in = _in + [width]*(depth-1)
            _out = [width]*depth
        elif isinstance(width, list):
            assert len(width) == depth, "Specifying width with a list should have `depth` elements"
            _in = _in + width[:-1]
            _out = width

        if isinstance(ksize, int):
            _ksizes = [ksize]*depth
        elif isinstance(ksize, list):
            assert len(
                ksize) == depth, "kernel size list should have 'depth' entries"
            _ksizes = ksize

        _activations = [activation]*depth
        # dont normalize in/out layers
        _norms = [norm_layer]*depth

        # Core processing layers, no norm at the first layer
        for lvl in range(depth):
            self.add_module(
                "conv{}".format(lvl),
                ConvModule(_in[lvl], _out[lvl], _ksizes[lvl], stride=_strides[lvl], pad=pad,
                           activation=_activations[lvl], norm_layer=_norms[lvl]))

    def forward(self, x):
        for m in self.children():
            x = m(x)
        return x


class ResidualBlock(nn.Module):
    """Basic residual block from [He 2015].

    <https://arxiv.org/pdf/1512.03385.pdf>

    Args:
      n_features(int): number of input/output channels.
      ksize(int): size of the convolution kernel (square).
      n_convs(int): number of convolutions in the non-skip path.
      activation(str): nonlinear activation function between convolutions.
      norm_layer(str): normalization to apply between the convolution modules.
    """

    def __init__(self, n_features, ksize=3, n_convs=2, activation=None, norm_layer=None):
        super(ResidualBlock, self).__init__()

        assert isinstance(n_features, int) and n_features > 0, \
            "Channels should be a positive integer got {}".format(n_features)
        assert isinstance(
            ksize, int) and ksize > 0, "Kernel size should be a positive integer got {}".format(ksize)
        assert isinstance(
            n_convs, int) and n_convs >= 2, "Number of convolutions should be at least 2."

        padding = (ksize - 1) // 2

        self.n_convs = n_convs
        self.convpath = th.nn.Sequential(
            ConvChain(n_features, ksize=ksize, width=n_features,
                      depth=n_convs-1, pad=True, activation=activation, norm_layer=norm_layer),
            ConvModule(n_features, n_features, ksize=ksize, stride=1, pad=True, activation=None, norm_layer=None)  # last layer has no activation
        )

        self.post_skip_activation = None
        if activation:
            self.post_skip_activation = _get_activation(activation)

    def forward(self, x):
        x = self.convpath(x) + x  # residual connection
        if self.post_skip_activation is not None:
            x = self.post_skip_activation(x)
        return x


class ResidualChain(nn.Module):
    """Linear chain of residual blocks.

    Args:
      n_features(int): number of input channels.
      ksize(int): size of the convolution kernel (square).
      depth(int): number of residual blocks
      convs_per_block(int): number of convolution per residual block
      activation(str): nonlinear activation function between convolutions.
      norm_layer(str): normalization to apply between the convolution modules.
    """

    def __init__(self, n_features, ksize=3, depth=3, convs_per_block=2,
                 activation="relu", norm_layer=None):
        super(ResidualChain, self).__init__()
        LOG.warning("ResidualChain has not been tested, beware!")

        assert isinstance(
            n_features, int) and n_features > 0, "Number of feature channels should be a positive integer"
        assert (isinstance(ksize, int) and ksize > 0) or isinstance(
            ksize, list), "Kernel size should be a positive integer or a list of integers"
        assert isinstance(
            depth, int) and depth > 0, "Depth should be a positive integer"

        # Core processing layers
        for lvl in range(depth):
            self.add_module(
                "resblock{}".format(lvl),
                ResidualBlock(n_features, ksize=ksize, 
                              n_convs=convs_per_block, activation=activation,
                              norm_layer=norm_layer))

    def forward(self, x):
        for m in self.children():
            x = m(x)
        return x


class UNet(nn.Module):
    """Simple UNet with downsampling and concat operations.

    Args:
      n_in(int): number of input channels.
      n_out(int): number of input channels.
      ksize(int): size of the convolution kernel (square).
      width(int): number of features channels in the first hidden layers.
      increase_factor(float): ratio of feature increase between scales.
      num_convs(int): number of conv layers per level
      num_levels(int): number of scales
      activation(str): nonlinear activation function between convolutions.
      norm_layer(str): normalization to apply between the convolution modules.
    """
    def _width(self, lvl):
        return min(self.base_width*self.increase_factor**(lvl), self.max_width)

    def __init__(self, n_in, n_out, ksize=3, base_width=64, max_width=512, increase_factor=2,
                 num_convs=1, num_levels=4, activation="relu", norm_layer=None,
                 interp_mode="bilinear"):
        super(UNet, self).__init__()

        self.increase_factor = increase_factor
        self.max_width = max_width
        self.base_width = base_width

        child = None
        lvl_in = []
        for lvl in range(num_levels-1, -1, -1):
            lvl_w = self._width(lvl)
            n_child_out = self._width(lvl)
            if lvl == 0:
                lvl_in = n_in
                lvl_out = n_out
            else:
                lvl_in = self._width(lvl-1)
                lvl_out = self._width(lvl-1)
                if lvl == num_levels-1:
                    n_child_out = 0

            u_lvl = UNet._UNetLevel(
                lvl_in, lvl_out, lvl_w, ksize, num_convs, activation,
                norm_layer, child=child, n_child_out=n_child_out,
                interp_mode=interp_mode)
            child = u_lvl
        self.top_level = u_lvl

    def forward(self, x):
        return self.top_level(x)

    class _UNetLevel(nn.Module):
        def __init__(self, n_in, n_out, width, ksize, num_convs, activation,
                     norm_layer, child=None, n_child_out=0, interp_mode="bilinear"):
            super(UNet._UNetLevel, self).__init__()
            self.left = ConvChain(n_in, ksize=ksize, width=width, depth=num_convs, pad=True,
                                  activation=activation, norm_layer=norm_layer)
            w = [width] * (num_convs-1) + [n_out]
            self.right = ConvChain(width + n_child_out, ksize=ksize, width=w, depth=num_convs, pad=True,
                                  activation=activation, norm_layer=norm_layer)
            self.child = child
            self.interp_mode = interp_mode

        def forward(self, x):
            left_features = self.left(x)
            if self.child is not None:
                ds = nn.functional.interpolate(
                    left_features, scale_factor=0.5, mode=self.interp_mode, align_corners=True)
                child_features = self.child(ds)
                us = nn.functional.interpolate(
                    child_features, size=left_features.shape[-2:],
                    mode=self.interp_mode, align_corners=True)

                # skip connection
                left_features = th.cat([left_features, us], 1)
            output = self.right(left_features)
            return output



# Helpers ---------------------------------------------------------------------

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
    valid = ["relu", "leaky_relu", "lrelu", "tanh", "sigmoid"]
    assert activation in valid, "activation should be one of {}".format(valid)
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "leaky_relu" or activation == "lrelu":
        return nn.LeakyReLU(inplace=True)
    if activation == "sigmoid":
        return nn.Sigmoid()
    if activation == "tanh":
        return nn.Tanh()
    return None


def _init_fc_or_conv(fc_conv, activation):
    gain = 1.0
    if activation is not None:
        gain = nn.init.calculate_gain(activation)
    nn.init.xavier_uniform_(fc_conv.weight, gain)
    if fc_conv.bias is not None:
        nn.init.constant_(fc_conv.bias, 0.0)

# -----------------------------------------------------------------------------

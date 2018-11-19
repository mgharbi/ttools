"""Common used neural networks operations."""

# TODO(mgharbi): maybe add a norm layer at the output if we specify an activation fn ?


from torch import nn


class FCModule(nn.Module):
    """Basic fully connected module with optional dropout.

    Args:
      n_in(int): number of input channels.
      n_out(int): number of output channels.
      activation(str): nonlinear activation function.
      dropout(float): dropout ratio if defined, default to None: no dropout.
    """

    def __init__(self, n_in, n_out, activation="relu", dropout=None):
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
      n_out(int): number of output channels.
      width(int or list of int): number of features channels in the intermediate layers.
      depth(int): number of layers
      activation(str): nonlinear activation function between convolutions.
      dropout(float or list of float): dropout ratio if defined, default to None: no dropout.
      out_activation(str): activation function applied to the output, defaults to linear (none).
    """

    def __init__(self, n_in, n_out, width=64, depth=3, activation="relu",
                 dropout=None, out_activation=None):
        super(FCChain, self).__init__()

        assert isinstance(
            n_in, int) and n_in > 0, "Input channels should be a positive integer"
        assert isinstance(
            n_out, int) and n_out > 0, "Output channels should be a positive integer"
        assert isinstance(
            depth, int) and depth > 0, "Depth should be a positive integer"
        assert isinstance(width, int) or isinstance(
            width, list), "Width should be a list or an int"

        _in = [n_in]
        _out = [n_out]

        if isinstance(width, int):
            _in = _in + [width]*(depth-1)
            _out = [width]*(depth-1) + _out
        elif isinstance(width, list):
            assert depth > 1 and len(
                width) == depth-1, "Needs at least two layers to specify width with a list. Do no specify out"
            _in = _in + width
            _out = width + _out

        _activations = [activation]*(depth-1) + [out_activation]

        if dropout is not None:
            assert isinstance(dropout, float) or isinstance(
                dropout, list), "Dropout should be a float or a list of floats"

        if dropout is None or isinstance(dropout, float):
            # dont do dropout on in/out layers
            _dropout = [None] + [dropout]*(depth-2) + [None]
        elif isinstance(dropout, list):
            assert depth > 2 and len(
                dropout) == depth-2, "Needs at least three layers to specify dropout with a list (do not specify in/out)."
            # dont do dropout on in/out layers
            _dropout = [None] + dropout + [None]

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
      stride(int): 
      pad(bool): if True, zero pad the convolutions to maintain a constant size.
      activation(str): nonlinear activation function between convolutions.
      norm_layer(str): normalization to apply between the convolution modules.
    """

    def __init__(self, n_in, n_out, ksize, stride=1, pad=True,
                 activation="relu", norm_layer=None):
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
      n_out(int): number of output channels.
      ksize(int or list of int): size of the convolution kernel (square).
      width(int or list of int): number of features channels in the intermediate layers.
      depth(int): number of layers
      strides(list of int): stride between kernels. If None, defaults to 1 for all.
      pad(bool): if True, zero pad the convolutions to maintain a constant size.
      activation(str): nonlinear activation function between convolutions.
      norm_layer(str): normalization to apply between the convolution modules.
      out_activation(str): activation function applied to the output, defaults to linear (none).
    """

    def __init__(self, n_in, n_out, ksize=3, width=64, depth=3, strides=None, pad=True,
                 activation="relu", norm_layer=None, out_activation=None):
        super(ConvChain, self).__init__()

        assert isinstance(
            n_in, int) and n_in > 0, "Input channels should be a positive integer"
        assert isinstance(
            n_out, int) and n_out > 0, "Output channels should be a positive integer"
        assert (isinstance(ksize, int) and ksize > 0) or isinstance(
            ksize, list), "Kernel size should be a positive integer or a list of integers"
        assert isinstance(
            depth, int) and depth > 0, "Depth should be a positive integer"
        assert isinstance(width, int) or isinstance(
            width, list), "Width should be a list or an int"

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
            assert depth > 1 and len(
                width) == depth-1, "Needs at least two layers to specify width with a list."
            _in = _in + width
            _out = width + _out

        if isinstance(ksize, int):
            _ksizes = [ksize]*depth
        elif isinstance(ksize, list):
            assert len(
                ksize) == depth, "kernel size list should have 'depth' entries"
            _ksizes = ksize

        _activations = [activation]*(depth-1) + [out_activation]
        # dont normalize in/out layers
        _norms = [None] + [norm_layer]*(depth-2) + [None]

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


class DownConvChain(ConvChain):
    """Chain of convolution layers with 2x spatial downsamplings.

    Args:
      n_in(int): number of input channels.
      n_out(int): number of output channels.
      ksize(int): size of the convolution kernel (square).
      base_width(int): number of features channels in the intermediate layers.
      increase_factor(float): multiplicative factor on feature size after downsampling.
      num_levels(int): number of downsampling levels.
      convs_per_level(int or list of ints): number of conv layers at each levels.
      pad(bool): if True, zero pad the convolutions to maintain a constant size.
      activation(str): nonlinear activation function between convolutions.
      norm_layer(str): normalization to apply between the convolution modules.
      out_activation(str): activation function applied to the output, defaults to linear (none).
    """

    def __init__(self, n_in, n_out, ksize=3, base_width=64, increase_factor=2.0,
                 num_levels=3, convs_per_level=2, pad=True,
                 activation="relu", norm_layer=None, out_activation=None):

        assert isinstance(
            num_levels, int) and num_levels > 0, "num_levels should be a positive integer"
        assert isinstance(convs_per_level, int) or isinstance(
            convs_per_level, list), "convs_per_level should be a positive integer or list of ints"
        assert isinstance(
            increase_factor, float) and increase_factor >= 1.0, "increase_factor should be a float >= 1.0"

        if (increase_factor, int):
            increase_factor = [increase_factor]*num_levels
        assert len(
            increase_factor) == num_levels, "increase_factor should have num_levels entries"

        depth = num_levels*convs_per_level + 1
        widths = []
        strides = []
        w = base_width
        for lvl in range(num_levels):
            for cv in range(convs_per_level):
                if cv == convs_per_level-1:
                    strides.append(2)
                else:
                    strides.append(1)
                widths.append(w)
            w = int(w*increase_factor[lvl])
        strides.append(1)

        super(DownConvChain, self).__init__(
            n_in, n_out, ksize=ksize, depth=depth, strides=strides, pad=pad,
            activation=activation, norm_layer=norm_layer,
            out_activation=out_activation)

# class ConvAutoencoder(nn.Module):
#   """Convolutionnal autoencoder with a spatially downsampled bottleneck.
#
#   Args:
#     n_in(int): number of input channels.
#     n_out(int): number of output channels.
#     ksize(int or list of int): size of the convolution kernel (square).
#     width(int or list of int): number of features channels in the intermediate layers.
#     depth(int): number of downsampling layers
#     strides(list of int): stride between kernels. If None, defaults to 1 for all.
#     pad(bool): if True, zero pad the convolutions to maintain a constant size.
#     activation(str): nonlinear activation function between convolutions.
#     norm_layer(str): normalization to apply between the convolution modules.
#     out_activation(str): activation function applied to the output, defaults to linear (none).
#   """
#   def __init__(self, n_in, n_out, ksize=3, width=64, depth=3, strides=None, pad=True,
#                activation="relu", norm_layer=None, out_activation=None):
#     super(ConvAutoencoder, self).__init__()
#
#     self.encoder = DownConvChain(n_in, n_bottleneck, ksize=ksize, width=e_widths,
#                              strides=e_strides, depth=e_depth, pad=pad,
#                              activation=activation, norm_layer=norm_layer,
#                              out_activation=activation)
#
#     self.decoder = UpConvChain()
#
#   def __init__(self, n_in, n_out, ksize=3, width=64, depth=3, strides=None, pad=True,
#                activation="relu", norm_layer=None, out_activation=None):

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decode(z)

    def forward(self, x):
        return self.decode(self.encode(x))


class UpConvChain(nn.Module):
    pass


class UNet(nn.Module):
    """
    """

    def __init__(self):
        pass

    class _UNetLevel(nn.Module):
        """
        """

        def __init__(self):
            pass

        def _encode(self):
            pass

        def _upsample(self):
            pass

        def forward(self, x):
            encoded = self.encode(x)
            if self.has_child():
                ds = self.downsample(encoded)
                lowres_features = self.child(ds)
                us = self.upsample(lowres_features)
                encoded = self.skip_connect(us, encoded)
            return self.decode(encoded)


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
    return None


def _init_fc_or_conv(fc_conv, activation):
    gain = 1.0
    if activation is not None:
        gain = nn.init.calculate_gain(activation)
    nn.init.xavier_uniform_(fc_conv.weight, gain)
    if fc_conv.bias is not None:
        nn.init.constant_(fc_conv.bias, 0.0)

"""Models."""
import torch as th
import numpy as np

import ttools
from ttools.modules import networks


LOG = ttools.get_logger(__name__)


__all__ = ["BasicModel"]


class BasicModel(th.nn.Module):
    """Dummy model."""
    def __init__(self, depth=2):
        super(BasicModel, self).__init__()

        assert depth > 1, "depth should be > 1"

        self.net = networks.ConvChain(3, width=16, ksize=3,
                                      depth=depth,
                                      activation='leaky_relu')
        self.regressor = networks.ConvModule(16, 3, ksize=1)

    def forward(self, x):
        return self.regressor(self.net(x))

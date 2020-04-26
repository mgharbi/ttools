import unittest

import torch as th
import torch.nn as nn

from ttools.modules import image_operators as imops

import imageio

from ttools import utils

class TestConvModule(unittest.TestCase):
    def setUp(self):
        pass

    def test_size(self):
        bs = 1
        c = 3
        h = 8
        w = h
        for scale in range(2, 8 + 1):
            in_ = th.rand(bs, c, h, w)
            op = imops.BilinearUpsampler(scale=scale, channels=c)
            out = op(in_)
            assert out.shape[2] == h*scale
            assert out.shape[3] == w*scale
        # imageio.imsave("bil_in.png", utils.tensor2image(in_))
        # imageio.imsave("bil_out.png", utils.tensor2image(out))

"""Helpers to debug pytorch code."""
import random
import string
import visdom

import torch as th


__all__ = ["tensor"]


_api = visdom.Visdom(port=8080, env="debug")


def _win(name):
    if name is None:
        name = ''.join([random.choice(string.ascii_letters) for n in range(16)])
    return name


def tensor(t, name):
    if len(t.shape) != 4:
        raise ValueError("Debug display needs 4D tensors")
    b, c, h, w = t.shape
    if c != 3:  # unroll channels
        t = t.view(b*c, 1, h, w)

    # normalize for display
    mini = t.min()
    maxi = t.max()

    if False:
        mu = t.mean()
        std = t.std()
        t = 0.5*((t-mu) / 2*std + 1)
    else:
        t = (t-mini) / (maxi - mini + 1e-12)

    opts = {
        "caption": "{} [{:.2f}, {:.2f}]".format(name, mini, maxi)
    }

    t = th.clamp(t, 0, 1)
    _api.images(t, win=_win(name), nrow=b, opts=opts)

"""Helpers to debug pytorch code."""
import random
import string
import visdom

import torch as th
import numpy as np


__all__ = ["tensor"]


_api = visdom.Visdom(port=8080, env="debug")


def _win(name):
    if name is None:
        name = ''.join([random.choice(string.ascii_letters) for n in range(16)])
    return name


def tensor(t, name, normalize=True, minmax=False):
    if isinstance(t, np.ndarray) and len(t.shape) == 3:
        t = th.from_numpy(t).permute(2, 0, 1).unsqueeze(0)
    elif len(t.shape) != 4:
        raise ValueError("Debug display needs 4D tensors")
    b, c, h, w = t.shape
    if c != 3:  # unroll channels
        t = t.view(b*c, 1, h, w)

    mini = t.min()
    maxi = t.max()

    # normalize for display
    if normalize:
        if minmax:
            t = (t-mini) / (maxi - mini + 1e-12)
        else:
            mu = t.mean()
            std = t.std()
            t = 0.5*((t-mu) / 2*std + 1)

    opts = {
        "caption": "{} [{:.2f}, {:.2f}]".format(name, mini, maxi)
    }

    t = th.clamp(t, 0, 1)
    _api.images(t, win=_win(name), nrow=b, opts=opts)


def scatter(x, y, name):
    if isinstance(x, np.ndarray):
        x = th.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = th.from_numpy(y)
    x = x.cpu().view(-1)
    y = y.cpu().view(-1)

    # opts = {
    #     "caption": "{} [{:.2f}, {:.2f}]".format(name, mini, maxi)
    # }

    xy = th.stack([x, y], 1)

    _api.scatter(xy, win=_win(name))

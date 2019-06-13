"""Helpers classes and functions."""

import torch as th
import numpy as np

class ExponentialMovingAverage(object):
    """Keyed tracker that maintains an exponential moving average for each key.

    Args:
      keys(list of str): keys to track.
      alpha(float): exponential smoothing factor (higher = smoother).
    """

    def __init__(self, keys, alpha=0.999):
        self._is_first_update = {k: True for k in keys}
        self._alpha = alpha
        self._values = {k: 0 for k in keys}

    def __getitem__(self, key):
        return self._values[key]

    def update(self, key, value):
        if self._is_first_update[key]:
            self._values[key] = value
            self._is_first_update[key] = False
        else:
            self._values[key] = self._values[key] * \
                self._alpha + value*(1.0-self._alpha)


class Averager(object):
    """Keeps track of running averages, for each key."""

    def __init__(self, keys):
        self.values = {k: 0.0 for k in keys}
        self.counts = {k: 0 for k in keys}

    def __getitem__(self, key):
        if self.counts[key] == 0:
            return 0.0
        return self.values[key] * 1.0/self.counts[key]

    def reset(self):
        for k in self.values.keys():
            self.values[k] = 0.0
            self.counts[k] = 0

    def update(self, key, value, count=1):
        self.values[key] += value*count
        self.counts[key] += count


def parse_params(plist):
    params = {}
    if plist is not None:
        for p in plist:
            k, v = p.split("=")
            if v.isdigit():
                v = int(v)
            elif v == "False":
                v = False
            elif v == "True":
                v = True
            params[k] = v
    return params


def tensor2image(t, normalize=False):
    """Converts an tensor image (4D tensor) to a numpy 8-bit array.
    Args:
        t(th.Tensor): input tensor with dimensions [bs, c, h, w], c=3, bs=1
        normalize(bool): if True, normalize the tensor's range to [0, 1] before clipping
    Returns:
        (np.array): [h, w, c] image in uint8 format, with c=3
    """

    assert len(t.shape) == 4, "expected 4D tensor, got %d dimensions" % len(t.shape)
    bs, c, h, w = t.shape

    assert bs == 1, "expected batch_size 1 tensor, got %d" % bs
    t = t.squeeze(0)

    assert c == 3, "expected tensor with 3 channels, got %d" % c

    if normalize:
        m = t.min()
        M = t.max()
        t = (t-m) / (M-m+1e-8)


    t = th.clamp(t.permute(1, 2, 0), 0, 1).cpu().detach().numpy()*255.0

    return t.astype(np.uint8)

# class Timer(object):
#     """A simple named timer context.
#
#     Usage:
#       with Timer("header_name"):
#         do_sth()
#     """
#
#     def __init__(self, header=""):
#         self.header = header
#         self.time = 0
#
#     def __enter__(self):
#         self.time = time.time()
#
#     def __exit__(self, tpye, value, traceback):
#         elapsed = (time.time()-self.time)*1000
#         print("{}, {:.1f}ms".format(self.header, elapsed))

"""Helpers classes and functions."""
import time

import torch as th
import numpy as np

import logging
try:
    import coloredlogs
    coloredlogs.install()
    HAS_COLORED_LOGS = True
except:
    HAS_COLORED_LOGS = False


__all__ = ["ExponentialMovingAverage", "Averager", "Timer", "tensor2image", "get_logger", "set_logger"]


def set_logger(debug=False):
    """Set the default logging level and log format.

    Args:
        debug(bool): if True, enable debug logs.
    """

    log_level = logging.INFO
    prefix = "[%(process)d] %(levelname)s %(name)s"
    suffix = " | %(message)s"
    if debug:
        log_level = logging.DEBUG
        prefix += " %(filename)s:%(lineno)s"
    if HAS_COLORED_LOGS:
        coloredlogs.install(
            level=log_level,
            format=prefix+suffix)
    else:
        logging.basicConfig(
            level=log_level,
            format=prefix+suffix)


def get_logger(name):
    """Get a named logger.

    Args:
        name(string): name of the logger
    """
    return logging.getLogger(name)




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
        if value is None:
            return
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
        if value is None:
            return
        self.values[key] += value*count
        self.counts[key] += count


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

    assert c == 3 or c == 1, "expected tensor with 1 or 3 channels, got %d" % c

    if normalize:
        m = t.min()
        M = t.max()
        t = (t-m) / (M-m+1e-8)


    t = th.clamp(t.permute(1, 2, 0), 0, 1).cpu().detach().numpy()*255.0

    return t.astype(np.uint8)


class Timer(object):
    """A simple timer context.

    Returns timing in ms

    Args:
        sync(bool): if True, synchronize CUDA kernels.

    """

    def __init__(self, sync=True):
        self._time = 0
        self.sync = sync
        self.elapsed = None

    def __enter__(self):
        th.cuda.synchronize()
        self._time = time.time()
        return self

    def __exit__(self, tpye, value, traceback):
        th.cuda.synchronize()
        self.elapsed = (time.time()-self._time)*1000

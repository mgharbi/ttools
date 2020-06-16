"""Helpers classes and functions."""
import time
import os
import seaborn as sns
import matplotlib.pyplot as plt

import torch as th
import numpy as np
import imageio

import logging

try:
    import coloredlogs
    coloredlogs.install()
    HAS_COLORED_LOGS = True
except ImportError:
    HAS_COLORED_LOGS = False

from . import database

__all__ = ["ExponentialMovingAverage", "Averager", "Timer",
           "tensor2image", "get_logger", "set_logger",
           "imread", "imsave", "get_logs", "plot_logs"]


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


def tensor2image(t, normalize=False, dtype=np.uint8):
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

    t = th.clamp(t.permute(1, 2, 0), 0, 1).cpu().detach().numpy()

    if dtype == np.uint8:
        return (255.0*t).astype(np.uint8)
    elif dtype == np.uint16:
        return ((2**16-1)*t).astype(np.uint16)
    else:
        raise ValueError("dtype %s not recognized" % dtype)


def imsave(path, im, dtype="uint8"):
    """Assumes [0,1] float images as input.

    im can be a [bs, c, h, w] torch tensor or
    a [h, w, 3], [h, w, 1] or [h, w] numpy array.
    """

    if dtype not in ["uint8", "uint16"]:
        raise ValueError("Image type should be `uint8` or `uint16`")
    if dtype == "uint8":
        dtype = np.uint8
        scale = 255
    else:
        dtype = np.uint16
        scale = 2**16 - 1

    if isinstance(im, th.Tensor):
        im = im.detach()
        if len(im.shape) == 4:  # bs, c, h, w
            if im.shape[0] != 1:
                raise ValueError("expected a single image, not a batch (got batch size %d)" % im.shape[0])
            im = im.squeeze(0)
        if len(im.shape) != 3 or im.shape[0] not in [1, 3]:
            raise ValueError("got invalid image size %s" % im.shape)
        im = im.permute(1, 2, 0)
        # squeeze grayscale
        im = im.squeeze(2)
        im = im.cpu().numpy()
    else:
        if len(im.shape) == 3 and im.shape[0] not in [1, 3]:
            raise ValueError("got invalid image size %s" % im.shape)

    im = (np.clip(im, 0, 1) * scale).astype(dtype)

    imageio.imwrite(path, im)


def imread(path):
    """Reads an 8- or 16-bits image at path.

    Args:
        path(str): path to image file.

    Returns:
        np.ndarray of the proper type.
    """
    im = np.array(imageio.imread(path))
    dtype = im.dtype

    im = th.from_numpy(im.astype(np.float32))
    if len(im.shape) == 2:  # gray
        im = im.unsqueeze(0)
    else:
        im = im.permute(2, 0, 1)

    if dtype == np.uint8:
        im  = im / 255.0
    elif dtype == np.uint16:
        im  = im / (2**16-1)
    return im


def get_logs(path):
    """Fetches logs from a .sqlite database."""
    dbs = []
    if os.path.isdir(path):
        # search for default logs
        for f in os.listdir(path):
            if os.path.splitext(f)[-1] != ".sqlite":
                continue
            dbs.append(os.path.join(path, f))
        if len(dbs) > 1:
            msg = "Found %d .sqlite databases, "
            "please specify the exact path" % len(dbs)
            raise RuntimeError(msg)
        path = dbs[0]
    else:
        db = path

    db = database.SQLiteDatabase(path)
    events = db.read_table("events")
    logs = db.read_table("logs")
    val_logs = db.read_table("val_logs")
    return events, logs, val_logs


def plot_logs(path, outpath, key="loss"):
    # TODO: add log-scale, annotations, smoothing
    """Plot logs from a .sqlite database."""
    events, logs, val_logs = get_logs(path)

    sns.set()
    plt.clf()
    if key in logs.keys():
        sns.lineplot(x="step", y=key, data=logs)
    if key in val_logs.keys():
        sns.lineplot(x="step", y=key, data=val_logs)
    plt.legend(["train", "val"])

    training_end = events[events["event"]=="training_end"]["step"].tolist()

    for e in training_end:
        plt.axvline(e, c=[0.4, 0.1, 0.2])

    plt.title("%s vs. optimization step", key)
    plt.savefig(outpath)


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

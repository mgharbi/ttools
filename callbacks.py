"""Callbacks that can be added to a model trainer's maiin loop."""

# TODO: frequency parameter for loggers

import abc
import logging
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
import visdom


LOG = logging.getLogger(__name__)


class Callback(object):
    """Base class for all training callbacks."""

    def __repr__(self):
        return self.__class__.__name__

    def __init__(self):
        super(Callback, self).__init__()
        self.epoch = 0
        self.batch = 0
        self.datasize = 0
        self.val_datasize = 0

    def training_start(self, dataloader):
        self.datasize = len(dataloader)

    def training_end(self):
        pass

    def epoch_start(self, epoch):
        """Hook to execute code when a new epoch starts."""
        self.epoch = epoch

    def epoch_end(self):
        """Hook to execute code when an epoch ends."""
        pass

    def validation_start(self, dataloader):
        self.val_datasize = len(dataloader)

    def validation_step(self, fwd_data, val_data):
        pass

    def validation_end(self, val_data):
        pass

    def batch_start(self, batch, batch_data):
        self.batch = batch

    def batch_end(self, batch_data, fwd_result, bwd_result):
        pass


# TODO
class ExperimentLoggerCallback(Callback):
    def __init__(self, fname, meta=None):
        super(ExperimentLoggerCallback, self).__init__()

    def training_start(self, dataloader):
        super(ExperimentLoggerCallback, self).training_start(dataloader)
        print("logging experiment with", self.datasize)

    def training_end(self):
        super(ExperimentLoggerCallback, self).training_end()
        print("end logging experiment", self.epoch, self.batch)


class WebsiteGeneratorCallback(Callback):
    """"""
    pass

# TODO


class CSVLoggingCallback(Callback):
    """A callback that logs scalar quantities to a .csv file."""
    pass

#   def __init__(self, directory, keys, filename="log.csv"):
#     if directory.startswith('~'):
#        directory = os.path.expanduser(directory)
#     os.makedirs(directory, exist_ok=True)
#     self.logfile = open(os.path.join(directory, filename), 'a')
#     self.writer = csv.DictWriter(self.logfile, keys)
#     # TODO: only write header once
#     # self.writer.writeheader()
#
#     # TODO: make sure filename is csv, add max entries and list
#
#     # TODO: add periodic flush, cycles thru multiple log files as write fill up
#
#   def __del__(self):
#     # graceful closing of file
#     if self.logfile:
#       self.logfile.close()
#
#   def log(self, **kwargs):
#     """Write a log entry from a dictionary"""
#     self.writer.writerow(kwargs)


class KeyedCallback(Callback):
    """An abstract Callback that performs the same action for all keys in a list.

    The keys (resp. val_keys) are used to access the backward_data (resp.
    validation_data) produced by a ModelInterface.

    Args:
      keys (list of str): list of keys whose values will be logged during training.
      val_keys (list of str): list of keys whose values will be logged during validation
    """

    def __init__(self, keys=["loss"], val_keys=None):
        super(KeyedCallback, self).__init__()
        self.keys = keys
        self.val_keys = val_keys
        if self.val_keys is None:
            self.val_keys = keys


class VisdomLoggingCallback(KeyedCallback):
    """A callback that logs scalar quantities to a visdom server.

    Args:
      keys (list of str): list of keys whose values will be logged during training.
      val_keys (list of str): list of keys whose values will be logged during validation
      port (int): Port of the Visdom server to log to.
      env (string): name of the Visdom environment to log to.
    """

    def __init__(self, keys=["loss"], val_keys=["loss"], port=8097, env="main"):
        super(VisdomLoggingCallback, self).__init__(
            keys=keys, val_keys=val_keys)
        self._api = visdom.Visdom(port=port, env=env)

        self._opts = {}

        # Cleanup previous plots and setup options
        for k in self.keys:
            if self._api.win_exists(k):
                self._api.close(k)
            self._opts[k] = {
                "legend": ["train", "val"], "title": k, "xlabel": "epoch", "ylabel": k}

    def batch_end(self, batch_data, fwd, bwd):
        super(VisdomLoggingCallback, self).batch_end(batch_data, fwd, bwd)

        t = self.batch / self.datasize + self.epoch

        for k in self.keys:
            self._api.line([bwd[k]], [t], update="append", win=k, name="train",
                           opts=self._opts[k])

    def validation_end(self, val_data):
        super(VisdomLoggingCallback, self).validation_end(val_data)
        t = self.epoch + 1
        for k in self.val_keys:
            self._api.line([val_data[k]], [t], update="append", win=k, name="val",
                           opts=self._opts[k])


class LoggingCallback(KeyedCallback):
    """A callback that logs scalar quantities to the console.

    Make sure python's logging level is at least info to see the console prints.

    Args:
      name (str): name of the logger
      keys (list of str): list of keys whose values will be logged during training.
      val_keys (list of str): list of keys whose values will be logged during
        validation
    """

    TABSTOPS = 2

    def __init__(self, name, keys=["loss"], val_keys=None):
        super(LoggingCallback, self).__init__(keys=keys, val_keys=val_keys)

        self.log = logging.getLogger(name)
        self.log.setLevel(logging.INFO)

        self.m_indent = 0

    def __print(self, s):
        self.log.info(self.m_indent*LoggingCallback.TABSTOPS*' ' + s)

    def __indent(self, n=1):
        self.m_indent += n

    def __unindent(self, n=1):
        self.m_indent = max(0, self.m_indent-n)

    def training_start(self, dataloader):
        super(LoggingCallback, self).training_start(dataloader)
        self.__print("Training start")

    def training_end(self):
        super(LoggingCallback, self).training_end()
        self.__print("Training ended at epoch {}".format(self.epoch + 1))

    def epoch_start(self, epoch):
        super(LoggingCallback, self).epoch_start(epoch)
        self.__print("-- Epoch {} ".format(self.epoch + 1) + "-"*12)

    def validation_start(self, dataloader):
        super(LoggingCallback, self).validation_start(dataloader)
        self.__indent()
        # self.__print("Validation {}".format(self.epoch))

    def validation_end(self, val_data):
        super(LoggingCallback, self).validation_end(val_data)
        s = "Validation {} | ".format(self.epoch + 1)
        for k in self.keys:
            s += "{} = {:.2f} ".format(k, val_data[k])
        self.__print(s)
        self.__unindent()

    def batch_end(self, batch_data, fwd, bwd_data):
        """Logs training advancement Epoch.Batch"""
        super(LoggingCallback, self).batch_end(batch_data, fwd, bwd_data)
        s = "{}.{} | ".format(self.epoch + 1, batch + 1)
        for k in self.keys:
            s += "{} = {:.2f} ".format(k, bwd_data[k])
        self.__print(s)


class ProgressBarCallback(KeyedCallback):
    TABSTOPS = 2

    def __init__(self, keys=["loss"], val_keys=None):
        super(ProgressBarCallback, self).__init__(keys=keys, val_keys=val_keys)
        self.pbar = None

    def training_start(self, dataloader):
        super(ProgressBarCallback, self).training_start(dataloader)
        print("Training start")

    def training_end(self):
        super(ProgressBarCallback, self).training_end()
        print("Training ends")

    def epoch_start(self, epoch):
        super(ProgressBarCallback, self).epoch_start(epoch)
        self.pbar = tqdm(total=self.datasize, unit=" batches",
                         desc="Epoch {}".format(self.epoch + 1))

    def epoch_end(self):
        super(ProgressBarCallback, self).epoch_end()
        self.pbar.close()
        self.pbar = None

    def validation_start(self, dataloader):
        super(ProgressBarCallback, self).validation_start(dataloader)
        print(" "*ProgressBarCallback.TABSTOPS + "Running validation...")

    def validation_end(self, val_data):
        super(ProgressBarCallback, self).validation_end(val_data)
        s = " "*ProgressBarCallback.TABSTOPS + "Validation {} | ".format(
            self.epoch + 1)
        for k in self.keys:
            s += "{} = {:.2f} ".format(k, val_data[k])
        print(s)

    def batch_end(self, batch_data, fwd, bwd_data):
        super(ProgressBarCallback, self).batch_end(batch_data, fwd, bwd_data)
        self.pbar.update(1)
        d = {k: bwd_data[k] for k in self.keys}
        self.pbar.set_postfix(d)


class CheckpointingCallback(Callback):
    """A callback that periodically saves model checkpoints to disk.

    Args:
      checkpointer (Checkpointer): actual checkpointer responsible for the I/O.
      interval (int, optional): minimum time in seconds between periodic checkpoints
        (within an epoch). There is not periodic checkpoint if this value is None.
      max_files (int, optional): maximum number of periodic checkpoints to keep on disk. Note:
        epoch checkpoints are never discarded.
    """

    PERIODIC_PREFIX = "periodic_"
    EPOCH_PREFIX = "epoch_"

    def __init__(self, checkpointer, interval=600, max_files=5, prefix=None):
        super(CheckpointingCallback, self).__init__()
        self.checkpointer = checkpointer
        self.interval = interval
        self.max_files = max_files

        self.last_checkpoint_time = time.time()

        self.prefix = ""
        if prefix is not None:
            self.prefix = prefix

    def epoch_end(self):
        """Save a checkpoint at the end of each epoch."""

        super(CheckpointingCallback, self).epoch_end()
        self.checkpointer.save("{}{}{}".format(self.prefix, CheckpointingCallback.EPOCH_PREFIX,
                                               self.epoch))

    def training_end(self):
        super(CheckpointingCallback, self).training_end()
        self.checkpointer.save("{}training_end".format(self.prefix))

    def batch_end(self, batch_data, fwd_result, bwd_result):
        """Save a periodic checkpoint if requested."""

        super(CheckpointingCallback, self).batch_end(
            batch_data, fwd_result, bwd_result)

        if self.interval is None:  # We skip periodic checkpoints
            return

        now = time.time()

        delta = now - self.last_checkpoint_time

        if delta < self.interval:  # last checkpoint is too recent
            return

        LOG.debug("Periodic checkpoint")

        # TODO: add epoch in extras
        filename = "{}{}{}".format(self.prefix, CheckpointingCallback.PERIODIC_PREFIX,
                                   time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
        self.checkpointer.save(filename)

        self.__purge_old_files()

    def __purge_old_files(self):
        """Delete checkpoints that are beyond the max to keep."""
        if self.max_files is None:
            return

        chkpts = self.checkpointer.sorted_checkpoints()
        p_chkpts = []
        for c in chkpts:
            if c.startswith(self.prefix + CheckpointingCallback.PERIODIC_PREFIX):
                p_chkpts.append(c)

        if len(p_chkpts) > self.max_files:
            for c in p_chkpts[self.max_files:]:
                LOG.debug("CheckpointingCallback deleting {}".format(c))
                self.checkpointer.delete(c)


class ImageDisplayCallback(Callback, abc.ABC):
    """Displays image periodically to a Visdom server.

    This is an abstract class, subclasses should implement the visualized_image
    method that synthesizes a [B, C, H, W] image to be visualized.

    Args:
      frequency(int): number of optimization steps between two updates
      port (int): Port of the Visdom server to log to.
      env (string): name of the Visdom environment to log to.
    """

    def __init__(self, frequency=100, port=8097, env="main"):
        super(ImageDisplayCallback, self).__init__()
        self.freq = frequency
        self._api = visdom.Visdom(port=port, env=env)
        self.step = 0

    @abc.abstractmethod
    def visualized_image(self, batch, fwd_result, bwd_result):
        pass

    def batch_end(self, batch, fwd_result, bwd_result):
        if self.step % self.freq != 0:
            self.step += 1
            return

        self.step = 0

        opts = {"caption": "Epoch {}, batch {}".format(self.epoch, self.batch)}

        viz = self.visualized_image(batch, fwd_result, bwd_result)
        self._api.images(viz, win="images", opts=opts)
        self.step += 1

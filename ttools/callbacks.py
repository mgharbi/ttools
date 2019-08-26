"""Callbacks that can be added to a model trainer's main loop."""

# TODO: implement experiment logger
# TODO: implement csv logger

import abc
import logging
import random
import string
import subprocess
import time

from tqdm import tqdm
from torchvision.utils import make_grid
import visdom

from .utils import ExponentialMovingAverage


__all__ = [
    "Callback",
    "CheckpointingCallback",
    "LoggingCallback",
    "ImageDisplayCallback",
    "ProgressBarCallback",
    "VisdomLoggingCallback",
]


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

    def validation_step(self, batch, fwd_data, val_data):
        pass

    def validation_end(self, val_data):
        pass

    def batch_start(self, batch, batch_data):
        self.batch = batch

    def batch_end(self, batch_data, fwd_result, bwd_result):
        pass


class KeyedCallback(Callback):
    """An abstract Callback that performs the same action for all keys in a list.

    The keys (resp. val_keys) are used to access the backward_data (resp.
    validation_data) produced by a ModelInterface.

    Args:
      keys (list of str): list of keys whose values will be logged during training. Defaults
        to ["loss"].
      val_keys (list of str): list of keys whose values will be logged during validation
    """

    def __init__(self, keys=None, val_keys=None, smoothing=0.99):
        super(KeyedCallback, self).__init__()
        if keys is None:
            self.keys = ["loss"]
        else:
            self.keys = keys

        if val_keys is None:
            self.val_keys = self.keys
        else:
            self.val_keys = val_keys

        self.ema = ExponentialMovingAverage(self.keys, alpha=smoothing)

    def batch_end(self, batch_data, fwd, bwd):
        for k in self.keys:
            self.ema.update(k, bwd[k])


class VisdomLoggingCallback(KeyedCallback):
    """A callback that logs scalar quantities to a visdom server.

    Args:
      keys (list of str): list of keys whose values will be logged during training.
      val_keys (list of str): list of keys whose values will be logged during validation
      frequency(int): number of steps between display updates.
      port (int): Port of the Visdom server to log to.
      env (string): name of the Visdom environment to log to.
    """

    def __init__(self, keys=None, val_keys=None, frequency=100, port=8097,
                 env="main", log=False, smoothing=0.99):
        super(VisdomLoggingCallback, self).__init__(
            keys=keys, val_keys=val_keys, smoothing=smoothing)
        self._api = visdom.Visdom(port=port, env=env)

        self._opts = {}

        # Cleanup previous plots and setup options
        all_keys = set(self.keys + self.val_keys)
        for k in list(all_keys):
            if self._api.win_exists(k):
                self._api.close(k)
            legend = []
            if k in self.keys:
                legend.append("train")
            if k in self.val_keys:
                legend.append("val")
            self._opts[k] = {
                "legend": legend, "title": k, "xlabel": "epoch", "ylabel": k}
            if log:
                self._opts[k]["ytype"] = "log"

        self._step = 0
        self.frequency = frequency

    def batch_end(self, batch_data, fwd, bwd):
        super(VisdomLoggingCallback, self).batch_end(batch_data, fwd, bwd)

        if self._step % self.frequency != 0:
            self._step += 1
            return
        self._step = 0

        t = self.batch / self.datasize + self.epoch

        for k in self.keys:
            self._api.line([self.ema[k]], [t], update="append", win=k, name="train",
                           opts=self._opts[k])

        self._step += 1

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

    def __init__(self, name, keys=None, val_keys=None, frequency=100, smoothing=0.99):
        super(LoggingCallback, self).__init__(keys=keys, val_keys=val_keys, smoothing=smoothing)

        self.log = logging.getLogger(name)
        self.log.setLevel(logging.INFO)

        self.m_indent = 0

        self._step = 0
        self.frequency = frequency

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

        if self._step % self.frequency != 0:
            self._step += 1
            return
        self._step = 0

        s = "{}.{} | ".format(self.epoch + 1, self.batch + 1)
        for k in self.keys:
            s += "{} = {:.2f} ".format(k, bwd_data[k])
        self.__print(s)

        self._step += 1


class ProgressBarCallback(KeyedCallback):
    """A progress bar optimization logger."""

    def __init__(self, keys=None, val_keys=None, smoothing=0.99):
        super(ProgressBarCallback, self).__init__(keys=keys, val_keys=val_keys, smoothing=smoothing)
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
        print("Running validation...")
        self.pbar = tqdm(total=len(dataloader), unit=" batches",
                         desc="Validation {}".format(self.epoch + 1))

    def validation_step(self, batch, fwd_data, val_data):
        self.pbar.update(1)

    def validation_end(self, val_data):
        super(ProgressBarCallback, self).validation_end(val_data)
        self.pbar.close()
        self.pbar = None
        s = " "*ProgressBarCallback.TABSTOPS + "Validation {} | ".format(
            self.epoch + 1)
        for k in self.val_keys:
            s += "{} = {:.2f} ".format(k, val_data[k])
        print(s)

    def batch_end(self, batch_data, fwd, bwd_data):
        super(ProgressBarCallback, self).batch_end(batch_data, fwd, bwd_data)
        d = {}
        for k in self.keys:
            d[k] = self.ema[k]
        self.pbar.update(1)
        self.pbar.set_postfix(d)

    TABSTOPS = 2


class CheckpointingCallback(Callback):
    """A callback that periodically saves model checkpoints to disk.

    Args:
      checkpointer (Checkpointer): actual checkpointer responsible for the I/O.
      start_epoch (int or None): index of the starting epoch (e.g. when resuming from a previous checkpoint).
      interval (int, optional): minimum time in seconds between periodic checkpoints
        (within an epoch). There is not periodic checkpoint if this value is None.
      max_files (int, optional): maximum number of periodic checkpoints to keep on disk. Note:
        epoch checkpoints are never discarded.
    """

    PERIODIC_PREFIX = "periodic_"
    EPOCH_PREFIX = "epoch_"

    def __init__(self, checkpointer, start_epoch=None, interval=600, max_files=5):
        super(CheckpointingCallback, self).__init__()
        self.checkpointer = checkpointer
        self.interval = interval
        self.max_files = max_files

        self.last_checkpoint_time = time.time()

        if start_epoch is None:
            self.start_epoch = 0
        else:
            self.start_epoch = start_epoch

    def get_extras(self):
        return {"epoch": self.epoch + self.start_epoch}

    def epoch_end(self):
        """Save a checkpoint at the end of each epoch."""

        super(CheckpointingCallback, self).epoch_end()
        path = "{}{}".format(CheckpointingCallback.EPOCH_PREFIX, self.epoch)
        self.checkpointer.save(path, extras=self.get_extras())
        self.__purge_old_files()

    def training_end(self):
        super(CheckpointingCallback, self).training_end()
        self.checkpointer.save("training_end", extras=self.get_extras())

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
        self.last_checkpoint_time = now

        filename = "{}{}".format(CheckpointingCallback.PERIODIC_PREFIX,
                                   time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
        self.checkpointer.save(filename, extras=self.get_extras())
        self.__purge_old_files()

    def __purge_old_files(self):
        """Delete checkpoints that are beyond the max to keep."""
        if self.max_files is None:
            return

        chkpts = self.checkpointer.sorted_checkpoints()
        p_chkpts = []
        e_chkpts = []
        for c in chkpts:
            if c.startswith(self.checkpointer.prefix + CheckpointingCallback.PERIODIC_PREFIX):
                p_chkpts.append(c)

            if c.startswith(self.checkpointer.prefix + CheckpointingCallback.EPOCH_PREFIX):
                e_chkpts.append(c)

        if len(p_chkpts) > self.max_files:
            for c in p_chkpts[self.max_files:]:
                LOG.debug("CheckpointingCallback deleting {}".format(c))
                self.checkpointer.delete(c)

        if len(e_chkpts) > self.max_files:
            for c in e_chkpts[self.max_files:]:
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

    def __init__(self, frequency=100, port=8097, env="main", win=None):
        super(ImageDisplayCallback, self).__init__()
        self.freq = frequency
        self._api = visdom.Visdom(port=port, env=env)
        self._step = 0
        if win is None:
            self.win = _random_string()
        else:
            self.win = win

    @abc.abstractmethod
    def visualized_image(self, batch, fwd_result):
        pass

    def caption(self, batch, fwd_result):
        return ""

    def batch_end(self, batch, fwd_result, bwd_result):
        if self._step % self.freq != 0:
            self._step += 1
            return

        self._step = 0

        caption = self.caption(batch, fwd_result)
        opts = {"caption": "Epoch {}, batch {}: {}".format(
            self.epoch, self.batch, caption)}

        viz = self.visualized_image(batch, fwd_result)
        self._api.images(viz, win=self.win, opts=opts)
        self._step += 1

    def validation_start(self, dataloader):
        super(ImageDisplayCallback, self).validation_start(dataloader)
        self.first_step = True

    def validation_step(self, batch, fwd_data, val_data):
        super(ImageDisplayCallback, self).validation_step(batch, fwd_data, val_data)
        if not self.first_step:
            return

        caption = self.caption(batch, fwd_data)
        opts = {"caption": "Validation {}, batch {}: {}".format(
            self.epoch, self.batch, caption)}

        viz = self.visualized_image(batch, fwd_data)
        self._api.images(viz, win=self.win+"_val", opts=opts)
        self.first_step = False


class ExperimentLoggerCallback(Callback):
    """A callback that logs experiment parameters in a log."""

    def __init__(self, fname, meta=None):
        super(ExperimentLoggerCallback, self).__init__()

    def training_start(self, dataloader):
        super(ExperimentLoggerCallback, self).training_start(dataloader)
        print("logging experiment with", self.datasize)

    def training_end(self):
        super(ExperimentLoggerCallback, self).training_end()
        print("end logging experiment", self.epoch, self.batch)

    def _get_commit(self):
       return subprocess.check_output(["git", "rev-parse", "HEAD"]) 


class CSVLoggingCallback(KeyedCallback):
    """A callback that logs scalar quantities to a .csv file.

    Format is:
        epoch, step, event, key, value
    """
    def __init__(self, fname, keys=None, val_keys=None, smoothing=0):
        super(CSVLoggingCallback, self).__init__(keys=keys, val_keys=val_keys, smoothing=smoothing)
        self.fname = fname
        self.fid = open(self.fname, 'w')

        self.fid.write("epoch, step, event, key, value\n")
        self.fid.write(",,logger_created,,\n")

        # open file, check last event

    def __del__(self):
        LOG.info("deleting csv logger")
        self.fid.write(",,logger_deleted,,\n")
        self.fid.close()

    def batch_end(self, batch_data, fwd, bwd_data):
        """Logs training advancement Batch"""
        super(CSVLoggingCallback, self).batch_end(batch_data, fwd, bwd_data)

        for k in self.keys:
            v = bwd_data[k]
            self.fid.write("%d,%d,batch_end,%s,%f\n" % (self.epoch, self.batch, k, v))

    def training_start(self, dataloader):
        super(CSVLoggingCallback, self).training_start(dataloader)
        self.fid.write(",,training_start,,\n")

    def training_end(self):
        super(CSVLoggingCallback, self).training_end()
        self.fid.write(",,training_end,,\n")


def _random_string(size=16):
    return ''.join([random.choice(string.ascii_letters) for i in range(size)])


# Tensorboard interface
class TensorBoardLoggingCallback(Callback):
    """A callback that logs scalar quantities to TensorBoard

    Args:
      keys (list of str): list of keys whose values will be logged during training.
      val_keys (list of str): list of keys whose values will be logged during validation
      frequency(int): number of steps between display updates.
      log_di (str)
    """

    def __init__(self, writer, val_writer, keys=None, val_keys=None, frequency=100, summary_type='scalar'):
        super(TensorBoardLoggingCallback, self).__init__()
        self.keys = keys
        self.val_keys = val_keys or self.keys
        self._writer = writer
        self._val_writer = val_writer
        self._step = 0
        self.frequency = frequency
        self.summary_type = summary_type

    def batch_end(self, batch_data, fwd, bwd):
        super(TensorBoardLoggingCallback, self).batch_end(batch_data, fwd, bwd)

        if self._step % self.frequency != 0:
            self._step += 1
            return
        self._step = 0

        t = self.batch + self.datasize * self.epoch

        for k in self.keys:
            if self.summary_type == 'scalar':
                self._writer.add_scalar(k, bwd[k], global_step=t)
            elif self.summary_type == 'histogram':
                self._writer.add_histogram(k, bwd[k], global_step=t)
        self._step += 1

    def validation_end(self, val_data):
        super(TensorBoardLoggingCallback, self).validation_end(val_data)


class TensorBoardImageDisplayCallback(Callback, abc.ABC):
    """Displays image periodically to TensorBoard.

    This is an abstract class, subclasses should implement the visualized_image
    method that synthesizes a [B, C, H, W] image to be visualized.

    Args:
      frequency(int): number of optimization steps between two updates
    """

    def __init__(self, writer, val_writer, frequency=100):
        super(TensorBoardImageDisplayCallback, self).__init__()
        self._writer = writer
        self._val_writer = val_writer
        self.freq = frequency
        self._step = 0

    @abc.abstractmethod
    def visualized_image(self, batch, fwd_result):
        pass

    @abc.abstractmethod
    def tag(self):
        pass

    def batch_end(self, batch, fwd_result, bwd_result):
        if self._step % self.freq != 0:
            self._step += 1
            return

        self._step = 0

        viz = self.visualized_image(batch, fwd_result)
        t = self.batch + self.datasize * self.epoch
        self._writer.add_image(self.tag(), make_grid(viz), t)
        self._step += 1

    def validation_start(self, dataloader):
        super(TensorBoardImageDisplayCallback, self).validation_start(dataloader)
        self.first_step = True

    def validation_step(self, batch, fwd_data, val_data):
        super(TensorBoardImageDisplayCallback, self).validation_step(batch, fwd_data, val_data)
        if not self.first_step:
            return

        viz = self.visualized_image(batch, fwd_data)
        t = self.datasize * (self.epoch+1)
        self._val_writer.add_image(self.tag(), make_grid(viz), t)
        self.first_step = False


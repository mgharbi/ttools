"""Utilities to train a model."""

from abc import ABCMeta, abstractmethod
import argparse
import os
import re
import signal

import torch as th

from .utils import get_logger


LOG = get_logger(__name__)


__all__ = ["ModelInterface", "Trainer", "Checkpointer", "BasicArgumentParser"]


class BasicArgumentParser(argparse.ArgumentParser):
    """A basic argument parser with commonly used training options."""

    def __init__(self):
        super(BasicArgumentParser, self).__init__()

        self.add_argument("--data", required=True, help="")
        self.add_argument("--checkpoint_dir", required=True,
                          help="Output directory where checkpoints are saved")

        self.add_argument("--init_from", help="path to a checkpoint from which to try and initialize the weights.")

        self.add_argument("--val_data", help="")

        self.add_argument("--lr", type=float, default=1e-4,
                          help="Learning rate for the optimizer")
        self.add_argument("--bs", type=int, default=4, help="Batch size")
        self.add_argument("--num_epochs", type=int,
                          help="Number of epochs to train for")
        self.add_argument("--num_worker_threads", default=4, type=int,
                          help="Number of threads that load data")

        self.add_argument("--experiment_log",
                          help="csv file in which we log our experiments")

        self.add_argument("--cuda", action="store_true",
                          dest="cuda", help="Force GPU")
        self.add_argument("--no-cuda", action="store_false",
                          dest="cuda", help="Force CPU")

        self.add_argument("--env", default="main", help="Visdom environment")
        self.add_argument("--port", default=8097, type=int,
                          help="Visdom server port")
        self.add_argument('--debug', dest="debug", action="store_true")

        self.set_defaults(cuda=th.cuda.is_available(), debug=False)


class ModelInterface(metaclass=ABCMeta):
    """An adapter to run or train a model."""

    def __init__(self):
        pass

    @abstractmethod
    def forward(self, batch):
        """Runs the model on a batch of data.

        Args:
          batch (dict): batch of data provided by a data pipeline.

        Returns:
          forward_data (dict): a dictionary of outputs
        """

        forward_data = {}

        return forward_data

    @abstractmethod
    def backward(self, batch, forward_data):
        """Computes gradients, take an optimizer step and update the model.

        Args:
          batch (dict): batch of data provided by a data pipeline.
          forward_data (dict): outputs from the forward pass

        Returns:
          backward_data (dict): a dictionary of outputs
        """

        backward_data = {}

        return backward_data

    def init_validation(self):
        """Initializes the quantities to be reported during validation.

        The default implementation is a no-op

        Returns:
          data (dict): initialized values
        """

        data = {}
        return data

    def update_validation(self, batch, fwd, running_data):
        """Updates the running val data using the current batch's forward output.

        The default implementation is a no-op

        Args:
          batch (dict): batch of data provided by a data pipeline.
          fwd (dict): data from one forward step in validation mode
          running_data (dict): current aggregates of the validation loop.

        Returns:
          updated_data (dict): initialized values
        """

        updated_data = {}
        return updated_data

    def finalize_validation(self, running_data):
        """Computes the final validation aggregates from the running data.

        The default implementation is a no-op

        Args:
          running_data (dict): current aggregates of the validation loop.

        Returns:
          validation_data (dict): initialized values
        """

        return running_data

    def __repr__(self):
        return self.__class__.__name__


class Trainer(object):
    """Implements a simple training loop with hooks for callbacks.

    Args:
      interface (ModelInterface): adapter to run forward and backward
        pass on the model being trained.

    Attributes:
      callbacks (list of Callbacks): hooks that will be called while training
        progresses.
    """

    def __init__(self, interface):
        super(Trainer, self).__init__()
        self.callbacks = []
        self.interface = interface
        LOG.debug("Creating {}".format(self))

        signal.signal(signal.SIGINT, self.interrupt_handler)

        self._keep_running = True

    def interrupt_handler(self, signo, frame):
        LOG.debug("interrupting run")
        self._keep_running = False

    def _stop(self):
        # Reset the run flag
        self._keep_running = True
        self.__training_end()

    def add_callback(self, callback):
        """Adds a callback to the list of training hooks."""
        LOG.debug("Adding callback {}".format(callback))
        self.callbacks.append(callback)

    def train(self, dataloader, num_epochs=None, val_dataloader=None):
        """Main training loop. This starts the training procedure.

        Args:
          dataloader (DataLoader): loader that yields training batches.
          num_epochs (int, optional): max number of epochs to run.
          val_dataloader (DataLoader, optional): loader that yields validation
            batches
        """
        self.__training_start(dataloader)
        epoch = 0
        while num_epochs is None or epoch < num_epochs:
            self.__epoch_start(epoch)
            for batch_idx, batch in enumerate(dataloader):
                if not self._keep_running:
                    self._stop()
                    return
                self.__batch_start(batch_idx, batch)
                fwd_result = self.__forward_step(batch)
                bwd_result = self.__backward_step(batch, fwd_result)
                self.__batch_end(batch, fwd_result, bwd_result)
            self.__epoch_end()

            # Validate
            if val_dataloader:
                with th.no_grad():
                    val_data = self.__validation_start(
                        val_dataloader)  # data interface adapter
                    for batch_idx, batch in enumerate(val_dataloader):
                        if not self._keep_running:
                            self._stop()
                            return
                        fwd_result = self.__forward_step(batch)
                        val_data = self.__validation_update(
                            batch, fwd_result, val_data)
                    self.__validation_end(val_data)

            epoch += 1

            if not self._keep_running:
                self._stop()
                return

        self._stop()

    def __repr__(self):
        return "Trainer({}, {} callbacks)".format(
            self.interface, len(self.callbacks))

    def __forward_step(self, batch):
        return self.interface.forward(batch)

    def __backward_step(self, batch, forward_data):
        return self.interface.backward(batch, forward_data)

    def __training_start(self, dataloader):
        for cb in self.callbacks:
            cb.training_start(dataloader)

    def __training_end(self):
        for cb in self.callbacks:
            cb.training_end()

    def __epoch_start(self, epoch_idx):
        for cb in self.callbacks:
            cb.epoch_start(epoch_idx)

    def __epoch_end(self):
        for cb in self.callbacks:
            cb.epoch_end()

    def __validation_start(self, dataloader):
        for cb in self.callbacks:
            cb.validation_start(dataloader)
        return self.interface.init_validation()

    def __validation_update(self, batch, fwd_data, val_data):
        for cb in self.callbacks:
            cb.validation_step(batch, fwd_data, val_data)
        return self.interface.update_validation(batch, fwd_data, val_data)

    def __validation_end(self, val_data):
        val_data = self.interface.finalize_validation(val_data)
        for cb in self.callbacks:
            cb.validation_end(val_data)

    def __batch_start(self, batch_idx, batch):
        for cb in self.callbacks:
            cb.batch_start(batch_idx, batch)

    def __batch_end(self, batch, fwd_result, bwd_result):
        for cb in self.callbacks:
            cb.batch_end(batch, fwd_result, bwd_result)


class Checkpointer(object):
    """Save and restore model and optimizer variables.

    Args:
      root (string): path to the root directory where the files are stored.
      model (torch.nn.Module):
      optimizer (torch.optimizer):
      meta (dict):
    """

    EXTENSION = ".pth"

    def __init__(self, root, model=None, meta=None, prefix=None):
        self.root = root
        self.model = model
        self.meta = meta

        LOG.debug(self)

        self.prefix = ""
        if prefix is not None:
            self.prefix = prefix

    def __repr__(self):
        return "Checkpointer with root at \"{}\"".format(self.root)

    def __path(self, path, prefix=None):
        if prefix is None:
            prefix = ""
        return os.path.join(self.root, prefix+os.path.splitext(path)[0] + ".pth")

    def save(self, path, extras=None):
        """Save model, metaparams and extras to relative path.

        Args:
          path (string): relative path to the file being saved (without extension).
          extras (dict): extra user-provided information to be saved with the model.
        """

        if self.model is None:
            model_state = None
        else:
            LOG.debug("Saving model state dict")
            model_state = self.model.state_dict()

        filename = self.__path(path, prefix=self.prefix)
        os.makedirs(self.root, exist_ok=True)
        th.save({'model': model_state,
                 'meta': self.meta,
                 'extras': extras,
                 }, filename)
        LOG.debug("Checkpoint saved to \"{}\"".format(filename))

    def try_and_init_from(self, path):
        LOG.info("Loading weights from foreign checkpoint {}".format(path))
        if not os.path.exists(path):
            raise ValueError("Checkpoint {} does not exist".format(path))
        chkpt = th.load(path)
        if "model" not in chkpt.keys() or chkpt["model"] is None:
            raise ValueError("{} has no model saved".format(path))
        mdl = chkpt["model"]
        for n, p in self.model.named_parameters():
            if n in mdl:
                p2 = mdl[n]
                if p2.shape != p.shape:
                    LOG.warning("Parameter {} ignored, checkpoint size does not match: {}, should be {}".format(n, p2.shape, p.shape))
                    continue
                LOG.debug("Parameter {} copied".format(n))
                p.data.copy_(p2)
            else:
                LOG.warning("Parameter {} ignored, not found in source checkpoint.".format(n))
        LOG.info("Weights loaded from foreign checkpoint {}".format(path))

    def load(self, path):
        """Loads a checkpoint, updates the model and returns extra data.

        Args:
          path (string): path to the checkpoint file, relative to the root dir.

        Returns:
          extras (dict): extra information passed by the user at save time.
          meta (dict): metaparameters of the model passed at save time.
        """

        filename = self.__path(path, prefix=None)
        chkpt = th.load(filename, map_location="cpu")  # TODO: check behavior

        if self.model is not None and chkpt["model"] is not None:
            LOG.debug("Loading model state dict")
            self.model.load_state_dict(chkpt["model"])

        LOG.debug("Loaded checkpoint \"{}\"".format(filename))
        return tuple(chkpt[k] for k in ["extras", "meta"])

    def load_latest(self):
        """Try to load the most recent checkpoint, skip failing files.

        Returns:
          extras (dict): extra information passed by the user at save time.
          meta (dict): metaparameters of the model passed at save time.
        """
        all_checkpoints = self.sorted_checkpoints()

        extras = None
        meta = None
        for f in all_checkpoints:
            try:
                extras, meta = self.load(f)
                return extras, meta
            except Exception as e:
                LOG.debug(
                    "Could not load checkpoint \"{}\", moving on ({}).".format(f, e))
        LOG.debug("No checkpoint found to load.")
        return extras, meta

    def sorted_checkpoints(self):
        """Get list of all checkpoints in root directory, sorted by creation date."""
        reg = re.compile(r"{}.*\{}".format(self.prefix, Checkpointer.EXTENSION))
        if not os.path.exists(self.root):
            all_checkpoints = []
        else:
            all_checkpoints = [f for f in os.listdir(
                self.root) if reg.match(f)]
        mtimes = []
        for f in all_checkpoints:
            mtimes.append(os.path.getmtime(os.path.join(self.root, f)))

        mf = sorted(zip(mtimes, all_checkpoints))
        chkpts = [m[1] for m in reversed(mf)]
        LOG.debug("Sorted checkpoints {}".format(chkpts))
        return chkpts

    def delete(self, path):
        """Delete checkpoint at path."""
        if path in self.sorted_checkpoints():
            os.remove(os.path.join(self.root, path))
        else:
            LOG.warning("Trying to delete a checkpoint that does not exists.")

    @staticmethod
    def load_meta(root, prefix=None):
        """Fetch model metadata without touching the saved parameters."""
        chkptr = Checkpointer(root, None, None, prefix=prefix)
        _, meta = chkptr.load_latest()
        return meta
